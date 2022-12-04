## Contributing to LQIT

> Modified from [MMCV](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md).

English | [简体中文](CONTRIBUTING_zh-CN.md)

All kinds of contributions are welcome, including but not limited to the following.

**Fix bug**

You can directly post a Pull Request to fix typo in code or documents

The steps to fix the bug of code implementation are as follows.

1. If the modification involve significant changes, you should create an issue first and describe the error information and how to trigger the bug. Other developers will discuss with you and propose an proper solution.

2. Posting a pull request after fixing the bug and adding corresponding unit test.

**New Feature or Enhancement**

1. If the modification involve significant changes, you should create an issue to discuss with our developers to propose an proper design.
2. Post a Pull Request after implementing the new feature or enhancement and add corresponding unit test.

**Document**

You can directly post a pull request to fix documents. If you want to add a document, you should first create an issue to check if it is reasonable.

### Workflow

If you're not familiar with Pull Request, don't worry! The following guidance will tell you how to create a Pull Request step by step. If you want to dive into the develop mode of Pull Request, you can refer to the [official documents](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

#### 1. Fork and clone

If you are posting a pull request for the first time, you should fork the LQIT by clicking the **Fork** button in the top right corner of the GitHub page, and the forked repositories will appear under your GitHub profile.

<img src="https://user-images.githubusercontent.com/48282753/205024875-fca32db5-af7f-47e5-a2a9-a25115d7cce1.png" width="1200">

Then, you can clone the repositories to local:

```shell
git clone git@github.com:{username}/lqit.git
```

After that, you should add official repository as the upstream repository

```bash
git remote add upstream git@github.com:BIGWangYuDong/lqit.git
```

Check whether remote repository has been added successfully by `git remote -v`

```bash
origin	git@github.com:{username}/lqit.git (fetch)
origin	git@github.com:{username}/lqit.git (push)
upstream	git@github.com:BIGWangYuDong/lqit.git (fetch)
upstream	git@github.com:BIGWangYuDong/lqit.git (push)
```

> Here's a brief introduction to origin and upstream. When we use "git clone", we create an "origin" remote by default, which points to the repository cloned from. As for "upstream", we add it ourselves to point to the target repository. Of course, if you don't like the name "upstream", you could name it as you wish. Usually, we'll push the code to "origin". If the pushed code conflicts with the latest code in official("upstream"), we should pull the latest code from upstream to resolve the conflicts, and then push to "origin" again. The posted Pull Request will be updated automatically.

#### 2. Configure pre-commit

You should configure [pre-commit](https://pre-commit.com/#intro) in the local development environment to make sure the code style matches that of LQIT. **Note**: The following code should be executed under the LQIT directory.

```shell
pip install -U pre-commit
pre-commit install
```

Check that pre-commit is configured successfully, and install the hooks defined in `.pre-commit-config.yaml`.

```shell
pre-commit run --all-files
```

<img src="https://user-images.githubusercontent.com/57566630/173660750-3df20a63-cb66-4d33-a986-1f643f1d8aaf.png" width="1200">

<img src="https://user-images.githubusercontent.com/57566630/202368856-0465a90d-8fce-4345-918e-67b8b9c82614.png" width="1200">

If the installation process is interrupted, you can repeatedly run `pre-commit run ... ` to continue the installation.

If the code does not conform to the code style specification, pre-commit will raise a warning and fixes some of the errors automatically.

<img src="https://user-images.githubusercontent.com/57566630/202369176-67642454-0025-4023-a095-263529107aa3.png" width="1200">

If we want to commit our code bypassing the pre-commit hook, we can use the `--no-verify` option(**only for temporarily commit**).

```shell
git commit -m "xxx" --no-verify
```

#### 3. Create a development branch

After configuring the pre-commit, we should create a branch based on the main branch to develop the new feature or fix the bug. The proposed branch name is `username/pr_name`

```shell
git checkout -b wyd/refactor_contributing_doc
```

In subsequent development, if the main branch of the local repository is behind the main branch of "upstream", we need to pull the upstream for synchronization, and then execute the above command:

```shell
git pull upstream main
```

#### 4. Commit the code and pass the unit test

- The committed code should pass through the unit test

  ```shell
  # Pass all unit tests
  pytest tests

  # Pass the unit test of runner
  pytest tests/test_runner/test_runner.py
  ```

#### 5. Push the code to remote

We could push the local commits to remote after passing through the check of unit test and pre-commit. You can associate the local branch with remote branch by adding `-u` option.

```shell
git push -u origin {branch_name}
```

This will allow you to use the `git push` command to push code directly next time, without having to specify a branch or the remote repository.

#### 6. Create a Pull Request

(1) Create a pull request in GitHub's Pull request interface

<img src="https://user-images.githubusercontent.com/48282753/205027324-bdcab8c4-ad83-45ce-9f1b-ee492740aff4.png" width="1200">

(2) Modify the PR description according to the guidelines so that other developers can better understand your changes

<img src="https://user-images.githubusercontent.com/48282753/205028509-1ac43d07-d33c-4918-a4cb-74ce2f33287f.png" width="1200">

Find more details about Pull Request description in [pull request guidelines](#pr-specs).

**note**

The Pull Request description should contain the reason for the change, the content of the change, and the impact of the change, and be associated with the relevant Issue (see [documentation](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue))

#### 7. Resolve conflicts

If your local branch conflicts with the latest master branch of "upstream", you'll need to resolove them. There are two ways to do this:

```shell
git fetch --all --prune
git rebase upstream/master
```

or

```shell
git fetch --all --prune
git merge upstream/master
```

If you are very good at handling conflicts, then you can use rebase to resolve conflicts, as this will keep your commit logs tidy. If you are not familiar with `rebase`, then you can use `merge` to resolve conflicts.

### Code style

#### Python

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](https://github.com/PyCQA/flake8): A wrapper around some linter tools.
- [isort](https://github.com/timothycrosley/isort): A Python utility to sort imports.
- [yapf](https://github.com/google/yapf): A formatter for Python files.
- [codespell](https://github.com/codespell-project/codespell): A Python utility to fix common misspellings in text files.
- [mdformat](https://github.com/executablebooks/mdformat): Mdformat is an opinionated Markdown formatter that can be used to enforce a consistent style in Markdown files.
- [docformatter](https://github.com/myint/docformatter): A formatter to format docstring.

Style configurations of yapf and isort can be found in [setup.cfg](./setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`, `markdown files`,
fixes `end-of-files`, `double-quoted-strings`, `python-encoding-pragma`, `mixed-line-ending`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](./.pre-commit-config.yaml).

### PR Specs

1. Use [pre-commit](https://pre-commit.com) hook to avoid issues of code style

2. One short-time branch should be matched with only one PR

3. Accomplish a detailed change in one PR. Avoid large PR

   - Bad: Support Faster R-CNN
   - Acceptable: Add a box head to Faster R-CNN
   - Good: Add a parameter to box head to support custom conv-layer number

4. Provide clear and significant commit message

5. Provide clear and meaningful PR description

   - Task name should be clarified in title. The general format is: \[Prefix\] Short description of the PR (Suffix)
   - Prefix: add new feature \[Feature\], fix bug \[Fix\], related to documents \[Docs\], in developing \[WIP\] (which will not be reviewed temporarily)
   - Introduce main changes, results and influences on other modules in short description
   - Associate related issues and pull requests with a milestone

6. If you introduce other third-party libraries, or borrow codes from third-party libraries, please confirm that their licenses are compatible with LQIT, and add `This code is inspired from http://` to the borrowed code
