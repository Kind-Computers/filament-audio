#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kind Computers, LLC.
set -e

echo "Running cargo clean..."
cargo clean

echo "Removing *.o files..."
find . -type f -name '*.o' -delete

echo "Removing *.d files..."
find . -type f -name '*.d' -delete

echo "Clean complete."
