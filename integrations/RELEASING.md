# Integration releases

The adapters are separate PyPI projects maintained in this repository. Their
release workflow is `.github/workflows/release-integrations.yml`. It is manually
started for one integration at a time; neither pushes, tags, merges, nor core
Trimwise releases automatically publish an adapter. The private PDF helper package
and both PDF applications are never published by this workflow.

## One-time account setup

Use [PyPI account publishing](https://pypi.org/manage/account/publishing/) to add two
**pending GitHub Trusted Publishers**, one for each row below. Pending publishers
can be registered before the workflow is pushed or any package is uploaded.

| PyPI project | GitHub owner | Repository | Workflow filename | Environment |
| --- | --- | --- | --- | --- |
| `langchain-trimwise` | `tenwritehq` | `trimwise` | `release-integrations.yml` | `pypi-langchain` |
| `llama-index-postprocessor-trimwise` | `tenwritehq` | `trimwise` | `release-integrations.yml` | `pypi-llama-index` |

Enter the workflow filename alone, without `.github/workflows/`. Match all values
exactly. If a project already exists under your account, add its publisher from
that project's Publishing settings instead. If someone else owns the name, stop;
a pending publisher does not establish ownership.

No PyPI API token or GitHub publishing secret is needed. Only the final publishing
job requests `id-token: write`; building and testing have read-only permissions.
The first successful upload creates the PyPI project and converts its pending
publisher into a normal publisher. Registering a pending publisher **does not
reserve a name**. See [PyPI's pending-publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

In [GitHub environment settings](https://github.com/tenwritehq/trimwise/settings/environments),
configure each environment with the tag policy and approval below:

| Environment | Allowed tag pattern | Required reviewer |
| --- | --- | --- |
| `pypi-langchain` | `langchain-v*` | `aakashH242` |
| `pypi-llama-index` | `llama-index-v*` | `aakashH242` |

Use **Selected branches and tags**, with one **Tag** rule and no branch rule.
Allow the initiating maintainer to approve their own deployment, so a sole
maintainer can release, but disable administrator bypass. Leave core Trimwise's
existing `pypi` environment and `release.yml` publisher unchanged.

## Check artifacts before release

Each integration has its own version in its `pyproject.toml`; versions need not
match one another or core Trimwise. Current initial versions are both `0.1.0`.
The workflow accepts stable `X.Y.Z` versions only.

From the repository root, for example:

```bash
python -m build integrations/langchain --outdir /tmp/trimwise-langchain-dist
python -m twine check --strict /tmp/trimwise-langchain-dist/*
```

Repeat for `integrations/llama-index` in a separate output directory. Use a fresh
virtual environment to install the wheel against published dependencies, run
`pip check`, run that integration's test suite with its own pytest configuration,
and run `examples/minimal.py`. Do not install the local core checkout into this
verification environment: the adapter must work with an available PyPI core.

The GitHub workflow performs those build, metadata, wheel-install, test, and
minimal-example checks before publishing. CI separately tests supported framework
and Python combinations. SDK source distributions include tests and the small
minimal example, not the PDF application environments or local results.

## Release after review and merge

Run these steps only after review and merge:

1. Merge the reviewed adapter and workflow changes into `main`, with CI passing.
2. Tag the intended merged commit using the selected adapter's version:
   `langchain-v0.1.0` or `llama-index-v0.1.0`. Push that tag when ready to release.
3. In GitHub Actions, open **Release framework integration to PyPI**. Choose the
   matching **tag** in the ref selector and select `langchain` or `llama-index`.
4. Start the workflow. It rejects branches, core `vX.Y.Z` tags, mismatched adapter
   tags/versions, and commits that are not ancestors of `origin/main`.
5. Review the build/test result and downloadable distributions. Approve the
   selected environment's deployment only when ready for the public PyPI upload.
6. Verify the package page and install the exact version in a clean environment.
   Then request ecosystem listing separately.

PyPI releases cannot be overwritten. If a published version needs a correction,
prepare a new version and tag. A failed upload should be inspected before retry;
the workflow deliberately does not silently skip existing distributions.

## Setup status

The two GitHub environments
are configured with the reviewer and tag restrictions above, with administrator
bypass disabled. Both PyPI pending publishers were registered and verified in
the `aakashh242` account on 2026-10-22, with the exact settings above.
Pending publishers become ordinary publishers on the first successful upload.
Use the release workflow above for every version, including the initial release.
