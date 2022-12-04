# 对LQIT的贡献

> Modified from [MMCV](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md).

[English](CONTRIBUTING.md) | 简体中文

我们欢迎任何类型的贡献，包括但不限于

**修复错误**

你可以直接提交一个拉取请求 (Pull Request, PR)，用于修改在代码或文档中的书写错误。

修复代码实现错误的步骤如下：

1. 如果提交的代码改动较大，建议先提交 issue，并正确描述 issue 的现象、原因和复现方式，讨论后确认修复方案。
2. 修复错误并补充相应的单元测试，提交拉取请求。

**新增功能或组件**

1. 如果新功能或模块涉及较大的代码改动，建议先提交 issue，确认功能的必要性。
2. 实现新增功能并添单元测试，提交拉取请求。

**文档补充**

修复文档可以直接提交拉取请求

添加文档或将文档翻译成其他语言步骤如下

1. 提交 issue，确认添加文档的必要性。
2. 添加文档，提交拉取请求。

### 拉取请求工作流

如果你对拉取请求不了解，没关系，接下来的内容将会从零开始，一步一步地指引你如何创建一个拉取请求。如果你想深入了解拉取请求的开发模式，可以参考 github [官方文档](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

#### 1. 复刻仓库

当你第一次提交拉取请求时，先复刻 LQIT 原代码库，点击 GitHub 页面右上角的 **Fork** 按钮，复刻后的代码库将会出现在你的 GitHub 个人主页下。

<img src="https://user-images.githubusercontent.com/48282753/205024875-fca32db5-af7f-47e5-a2a9-a25115d7cce1.png" width="1200">

然后，将代码克隆到本地

```shell
git clone git@github.com:{username}/lqit.git
```

之后， 添加原代码库为上游代码库

```bash
git remote add upstream git@github.com:BIGWangYuDong/lqit.git
```

检查 remote 是否添加成功，在终端输入 `git remote -v`

```bash
origin	git@github.com:{username}/lqit.git (fetch)
origin	git@github.com:{username}/lqit.git (push)
upstream	git@github.com:BIGWangYuDong/lqit.git (fetch)
upstream	git@github.com:BIGWangYuDong/lqit.git (push)
```

> 这里对 origin 和 upstream 进行一个简单的介绍，当我们使用 git clone 来克隆代码时，
> 会默认创建一个 origin 的 remote，它指向我们克隆的代码库地址，而 upstream 则是我们自己添加的，
> 用来指向原始代码库地址。当然如果你不喜欢他叫 upstream，也可以自己修改.
> 我们通常向 origin 提交代码（即 fork 下来的远程仓库），
> 然后向 upstream 提交一个 pull request。如果提交的代码和最新的代码发生冲突，
> 再从 upstream 拉取最新的代码，和本地分支解决冲突，再提交到 origin。

#### 2. 配置 pre-commit

在本地开发环境中，你需要配置 [pre-commit](https://pre-commit.com/#intro) ，以确保代码风格统一。  **注意**: 以下代码需要在 LQIT 目录下执行

```shell
pip install -U pre-commit
pre-commit install
```

检查 pre-commit 是否配置成功，并安装 `.pre-commit-config.yaml`中的钩子：

```shell
pre-commit run --all-files
```

<img src="https://user-images.githubusercontent.com/57566630/173660750-3df20a63-cb66-4d33-a986-1f643f1d8aaf.png" width="1200">

<img src="https://user-images.githubusercontent.com/57566630/202368856-0465a90d-8fce-4345-918e-67b8b9c82614.png" width="1200">

如果安装过程被中断，可以重复执行 `pre-commit run ...` 继续安装。

如果提交的代码不符合代码风格规范，pre-commit 会发出警告，并自动修复部分错误。

<img src="https://user-images.githubusercontent.com/57566630/202369176-67642454-0025-4023-a095-263529107aa3.png" width="1200">

如果我们想临时绕开 pre-commit 的检查提交一次代码，可以在 `git commit` 时加上 `--no-verify`（需要保证最后推送至远程仓库的代码能够通过 pre-commit 检查）。

```shell
git commit -m "xxx" --no-verify
```

#### 3. 创建开发分支

安装完 pre-commit 之后，我们需要基于 master 创建开发分支，建议的分支命名规则为 `username/pr_name`。

```shell
git checkout -b wyd/refactor_contributing_doc
```

在后续的开发中，如果本地仓库的 master 分支落后于 upstream 的 master 分支，我们需要先拉取 upstream 的代码进行同步，再执行上面的命令：

```shell
git pull upstream main
```

#### 4. 提交代码并在本地通过单元测试

- 提交的代码需要通过单元测试。

  ```shell
  # 通过所有单元测试
  pytest tests

  # 我们需要保证提交的代码能够通过修改模块的单元测试，以 runner 为例
  pytest tests/test_runner/test_runner.py
  ```

#### 5. 推送代码到远端

代码通过单元测试和 pre-commit 检查后，将代码推送到远程仓库，如果是第一次推送，可以在 `git push` 后加上 `-u` 参数以关联远程分支

```shell
git push -u origin {branch_name}
```

这样下次就可以直接使用 `git push` 命令推送代码了，而无需指定分支和远程仓库。

#### 6. 提交拉取请求（PR）

(1) 在 GitHub 的 Pull request 界面创建拉取请求

<img src="https://user-images.githubusercontent.com/48282753/205027324-bdcab8c4-ad83-45ce-9f1b-ee492740aff4.png" width="1200">

(2) 根据指引修改 PR 描述，以便于其他开发者更好地理解你的修改

<img src="https://user-images.githubusercontent.com/48282753/205028509-1ac43d07-d33c-4918-a4cb-74ce2f33287f.png" width="1200">

描述规范详见[拉取请求规范](#拉取请求规范)

**注意事项**

PR 描述应该包含修改理由、修改内容以及修改后带来的影响，并关联相关 Issue（具体方式见[文档](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)）

#### 7. 解决冲突

随着时间的推移，我们的代码库会不断更新，这时候，如果你的 PR 与主分支存在冲突，你需要解决冲突，解决冲突的方式有两种：

```shell
git fetch --all --prune
git rebase upstream/master
```

或者

```shell
git fetch --all --prune
git merge upstream/master
```

如果你非常善于处理冲突，那么可以使用 rebase 的方式来解决冲突，因为这能够保证你的 commit log 的整洁。如果你不太熟悉 `rebase` 的使用，那么可以使用 `merge` 的方式来解决冲突。

### 代码风格

#### Python

[PEP8](https://www.python.org/dev/peps/pep-0008/) 作为 LQIT 首选的代码规范.

我们使用以下工具检查和格式化代码：

- [flake8](https://github.com/PyCQA/flake8): Python 官方发布的代码规范检查工具，是多个检查工具的封装
- [isort](https://github.com/timothycrosley/isort): 自动调整模块导入顺序的工具
- [yapf](https://github.com/google/yapf): Google 发布的代码规范检查工具
- [codespell](https://github.com/codespell-project/codespell): 检查单词拼写是否有误
- [mdformat](https://github.com/executablebooks/mdformat): 检查 markdown 文件的工具
- [docformatter](https://github.com/myint/docformatter): 格式化 docstring 的工具

yapf 和 isort 的配置可以在 [setup.cfg](./setup.cfg) 找到

通过配置 [pre-commit hook](https://pre-commit.com/) ，我们可以在提交代码时自动检查和格式化 `flake8`、`yapf`、`isort`、`trailing whitespaces`、`markdown files`，修复 `end-of-files`、`double-quoted-strings`、`python-encoding-pragma`、`mixed-line-ending`，调整 `requirments.txt` 的包顺序。
pre-commit 钩子的配置可以在 [.pre-commit-config](./.pre-commit-config.yaml) 找到。

### 拉取请求规范

1. 使用 [pre-commit hook](https://pre-commit.com)，尽量减少代码风格相关问题

2. 一个`拉取请求`对应一个短期分支

3. 粒度要细，一个`拉取请求`只做一件事情，避免超大的`拉取请求`

   - 不好的: 实现 Faster R-CNN
   - 可以接受的: 给 Faster R-CNN 添加一个 box head
   - 好的: 给 box head 增加一个参数来支持自定义的 conv 层数

4. 每次 Commit 时需要提供清晰且有意义 commit 信息

5. 提供清晰且有意义的`拉取请求`描述

   - 标题写明白任务名称，一般格式:\[Prefix\] Short description of the pull request (Suffix)
   - prefix: 新增功能 \[Feature\], 修 bug \[Fix\], 文档相关 \[Docs\], 开发中 \[WIP\] (暂时不会被review)
   - 描述里介绍`拉取请求`的主要修改内容，结果，以及对其他部分的影响, 参考`拉取请求`模板
   - 关联相关的`议题` (issue) 和其他`拉取请求`

6. 如果引入了其他三方库，或借鉴了三方库的代码，请确认他们的许可证和 LQIT 兼容，并在借鉴的代码上补充 `This code is inspired from http://`
