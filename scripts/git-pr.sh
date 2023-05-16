#!/bin/bash
MASTER_BRANCH="v1.6"

# check if in master branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" == MASTER_BRANCH ]; then
  echo "Please run 'git pr' from a topic branch."
  exit 1
fi

# check if there are any uncommitted changes
git_status=$(git status --porcelain)

if [ -n "$git_status" ]; then
  read -p "Uncommitted changes found. Commit before running 'git pr'? (y/n) " ans
  if [ "$ans" = "y" ]; then
    git commit -m -a "Automatic commit for git-pr"
  else
    echo "Please commit or stash changes before running 'git pr'."
    exit 1
  fi
fi

# Merging master
echo "Merging master..."
git merge origin/$MASTER_BRANCH

# Checking pylint, xcxc, pytest without touching git
PRE_GIT_CHECK=$(find . -name pre-git-check.sh)
if test -f "$PRE_GIT_CHECK"; then
  $PRE_GIT_CHECK
  if [ $? -ne 0 ]; then
    echo "pre-git-check.sh failed. Exiting."
    exit 1
  fi
else
  echo "Missing pre-git-check.sh. Exiting."
  exit 1
fi

# create a new branch from current branch and reset to master
echo "Creating and switching to new topic branch..."
git_user=$(git config user.email | cut -d'@' -f1)
branch_name="${git_user}-git-pr-$RANDOM-$RANDOM"
git checkout -b $branch_name
git reset --soft origin/$MASTER_BRANCH

# Verify that a commit message was added
echo "Verifying commit message..."
if ! git commit -a ; then
    echo "Commit message is empty. Exiting."
    exit 1
fi

# Push the topic branch to origin
echo "Pushing topic branch to origin..."
git push -u origin $branch_name

# Generate a Github pull request (just the url, not actually making a PR)
echo "Generating Github pull request..."
pull_request_url="https://github.com/CarperAI/nmmo-environment/compare/$MASTER_BRANCH...CarperAI:nmmo-environment:$branch_name?expand=1"

echo "Pull request URL: $pull_request_url"
