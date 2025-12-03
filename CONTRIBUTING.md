# How to Contribute

We would love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Development Environment Setup

### Installing Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to automatically run code quality checks before each commit. Pre-commit hooks help maintain consistent code style and catch common issues early.

#### Installation

1. Install pre-commit (if not already installed):

```bash
pip install pre-commit
```

2. Install the git hook scripts:

```bash
pre-commit install
```

#### What Gets Checked

The pre-commit hooks will automatically run:

- **pyink**: Google's Python code formatter (enforces 80-char line length, 2-space indentation)
- **pylint**: Python linter following Google's style guide
- **Standard checks**: trailing whitespace, file endings, YAML/JSON validation, merge conflicts, etc.

#### Usage

Once installed, the hooks run automatically on `git commit`. If any hook fails:

1. Review the errors/warnings
2. Fix the issues (some hooks auto-fix formatting)
3. Stage the changes: `git add .`
4. Commit again: `git commit`

#### Running Manually

To run pre-commit hooks on all files without committing:

```bash
pre-commit run --all-files
```

To run a specific hook:

```bash
pre-commit run pyink --all-files
pre-commit run pylint --all-files
```

#### Skipping Hooks (Not Recommended)

In rare cases where you need to skip hooks:

```bash
git commit --no-verify
```

**Note**: CI checks will still run on pull requests, so it's best to fix issues locally.

## Contribution process

### Code Reviews

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.
