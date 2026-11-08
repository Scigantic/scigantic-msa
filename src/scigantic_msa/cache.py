"""Local cache on disk.

Layout under the cache directory (``SCIGANTIC_MSA_CACHE``, else
``$XDG_CACHE_HOME/scigantic-msa``, else ``~/.cache/scigantic-msa``)::

    alignments/<aa>/<seq_sha256>/<backend>--<dbkey>.msa.gz

Writes go to a temporary file in the same directory and are renamed into
place, so a reader never sees half a file. A per-key lock file stops two
parallel jobs from computing the same alignment twice.

On Windows the default is ``%LOCALAPPDATA%\\scigantic-msa\\cache``.
``scigantic-msa cache`` prints the directory in use.

Files are written 0600 and the directories under the cache directory are
0700, so other users of a shared machine cannot list which sequences were
searched. An older, looser ``alignments`` folder is tightened the first
time the cache is used, even only to read it, and the folders under it
when an entry is written.

Delete the directory to forget every alignment. Nothing in it is sent
anywhere.
"""
from __future__ import annotations

import errno
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .alignment import Alignment

__all__ = ["cache_dir", "atomic_write", "private_dirs", "FileLock", "AlignmentCache"]


def _absolute_env(name: str) -> Optional[str]:
    """The value of ``name`` when it is an absolute path, else None. The XDG
    Base Directory spec says a relative path there is invalid and must be
    ignored; used, it would put the cache under whatever folder a run starts
    in (a literal ``~/xdg`` would make a folder named ``~`` there)."""
    value = os.environ.get(name)
    return value if value and os.path.isabs(value) else None


def cache_dir() -> Path:
    env = os.environ.get("SCIGANTIC_MSA_CACHE")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = _absolute_env("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "scigantic-msa" / "cache"
    xdg = _absolute_env("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "scigantic-msa"


def _owner_only(d: Path) -> None:
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass  # not ours to change, or a file system without modes


def private_dirs(path: Path, base: Path) -> None:
    """Make directory ``path`` and every directory between ``base`` and it,
    mode 0700, tightening any that exist with looser modes. ``base`` itself
    is made 0700 only when it is created here: a cache directory the user
    chose and made keeps the mode they gave it."""
    if not base.is_dir():
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        _owner_only(base)
    d = base
    for part in path.relative_to(base).parts:
        d = d / part
        try:
            d.mkdir(mode=0o700)
        except FileExistsError:
            if not d.is_dir():
                raise
        _owner_only(d)


def atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class LockTimeout(TimeoutError):
    pass


class FileLock:
    """Exclusive lock on ``<path>.lock`` that works across processes.

    Uses fcntl on POSIX and msvcrt on Windows. The lock is released when the
    holder exits, even if it crashes, because the OS drops it with the file
    descriptor. So a ``.lock`` file left by a crashed run is never held: it
    is taken like any other and removed on release.

    The file is removed on release while still locked. A waiter that opened
    it before the removal checks after locking that its descriptor is still
    the file at ``path``, and starts again if not, so two processes never
    hold different files for the same key.
    """

    def __init__(self, path: Path, timeout: float = 4 * 3600, poll: float = 0.5,
                 make_dir: Optional[Callable[[], None]] = None):
        self.path = Path(str(path) + ".lock")
        self.timeout = timeout
        self.poll = poll
        self._make_dir = make_dir
        self._fh: Optional[IO[bytes]] = None

    def acquire(self) -> None:
        if self._make_dir is not None:
            self._make_dir()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            fh = open(self.path, "a+b")
            try:
                locked = self._try_lock(fh)
            except OSError:
                fh.close()
                raise
            if locked:
                if self._still_ours(fh):
                    self._fh = fh
                    return
                # The holder removed the file after we opened it; lock the new one.
                fh.close()
                continue
            fh.close()
            if time.monotonic() > deadline:
                raise LockTimeout(f"timed out waiting for {self.path}")
            time.sleep(self.poll)

    @staticmethod
    def _try_lock(fh: IO[bytes]) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK, errno.EDEADLK):
                return False
            raise

    def _still_ours(self, fh: IO[bytes]) -> bool:
        if sys.platform == "win32":
            return True  # an open file cannot be removed there
        try:
            a, b = os.fstat(fh.fileno()), os.stat(self.path)
        except OSError:
            return False
        return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                # Remove while still locked, so nobody can lock this file
                # after it is gone from the directory and think it is theirs.
                try:
                    os.unlink(self.path)
                except OSError:
                    pass
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
        if sys.platform == "win32":
            try:
                os.unlink(self.path)
            except OSError:
                pass  # another process has it open; it removes it later

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _safe(part: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in part)[:80]


class AlignmentCache:
    """MSAs this machine computed or downloaded, keyed by seq_sha256, backend
    and database versions."""

    def __init__(self, root: Optional[Path] = None):
        self.root = (root or cache_dir()) / "alignments"
        self._tightened = False

    def _tighten(self) -> None:
        """Make ``alignments/`` 0700 once per cache object, on first use,
        whether to read or write. A cache made by 0.2.7 or older has it
        0755, and a run that only reads would otherwise leave it so; with it
        0700 nobody else can list or enter the folders under it."""
        if self._tightened:
            return
        self._tightened = True
        try:
            if os.stat(self.root).st_mode & 0o077:
                _owner_only(self.root)
        except OSError:
            pass  # not there yet, or not ours

    def _dir(self, sha: str) -> Path:
        return self.root / sha[:2] / sha

    def path(self, sha: str, backend: str, dbkey: str) -> Path:
        return self._dir(sha) / f"{_safe(backend)}--{_safe(dbkey)}.msa.gz"

    def _make_dir(self, sha: str) -> None:
        private_dirs(self._dir(sha), self.root.parent)

    def lock(self, sha: str, backend: str, dbkey: str, timeout: float = 4 * 3600) -> FileLock:
        return FileLock(self.path(sha, backend, dbkey), timeout=timeout, make_dir=lambda: self._make_dir(sha))

    def entries(self, sha: str) -> List[Path]:
        self._tighten()
        d = self._dir(sha)
        if not d.is_dir():
            return []
        files = [p for p in d.iterdir() if p.name.endswith(".msa.gz") and not p.name.startswith(".")]
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    def get(self, sha: str, backend: Optional[str] = None, dbkey: Optional[str] = None) -> "Optional[Alignment]":
        from .a3m import A3MError
        from .alignment import Alignment

        self._tighten()
        if backend is not None and dbkey is not None:
            candidates = [self.path(sha, backend, dbkey)]
        else:
            candidates = self.entries(sha)
            if backend is not None:
                candidates = [p for p in candidates if p.name.startswith(_safe(backend) + "--")]
        for p in candidates:
            try:
                data = p.read_bytes()
            except OSError:
                continue
            try:
                msa = Alignment.from_bundle(data, source="cache")
            except (A3MError, ValueError):
                # A corrupt entry is removed so it cannot shadow a good one.
                try:
                    p.unlink()
                except OSError:
                    pass
                continue
            if msa.seq_sha256 != sha:
                continue
            # The path is kept on the object, not in provenance, so that a
            # cached result's provenance is the fresh one's and no local
            # path reaches the attribution file next to the model input.
            msa.cache_path = str(p)
            return msa
        return None

    def put(self, msa: "Alignment", backend: str, dbkey: str) -> Path:
        p = self.path(msa.seq_sha256, backend, dbkey)
        self._make_dir(msa.seq_sha256)
        atomic_write(p, msa.to_bundle(), mode=0o600)
        return p

    def remove(self, sha: str) -> int:
        n = 0
        for p in self.entries(sha):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n
