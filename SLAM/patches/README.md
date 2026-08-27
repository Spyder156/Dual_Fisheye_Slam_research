# Patches

Our modifications to upstream repos, as **diffs** against the pinned checkout —
not file copies. Copies drift silently against upstream; a patch fails loudly.

```
<repo>/<repo>.patch      git diff of tracked files
<repo>/new_files/        files we ADDED (git diff does not carry them)
```

Regenerate:  `git -C SLAM/third_party/<REPO> diff > SLAM/patches/<name>/<name>.patch`
Apply:       `git -C SLAM/third_party/<REPO> apply SLAM/patches/<name>/<name>.patch`
             then copy `new_files/` into the repo.
