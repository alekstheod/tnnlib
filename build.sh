#!/usr/bin/env bash

# Use local Bazelisk with Bazel 7
export PATH="$HOME/bin:$PATH"
export USE_BAZEL_VERSION=7.3.1

bazelisk --host_jvm_args=-Xmx3g --host_jvm_args=-Xms512m test \
	--jobs=2 \
	--config=asan //... \
	--test_tag_filters="-manual,-openCL" \
	--keep_going \
	--test_output=all
