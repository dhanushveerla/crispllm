"""EdgeMem AST Context Slicer — `edgemem.slicer`

Adapts Aider's RepoMap algorithm (Apache-2.0) to produce a compact,
ranked, token-budgeted repository map from multi-language source trees.

Attribution
-----------
Portions of the PageRank graph-building and TreeContext formatting strategy
are adapted from the Aider project (https://github.com/Aider-AI/aider),
Copyright (c) 2023 Paul Gauthier, licensed under the Apache License, Version 2.0.
See <http://www.apache.org/licenses/LICENSE-2.0> for the full license text.

The dependency footprint has been reduced from ~100 packages (full aider-chat)
down to five packages: tree-sitter-language-pack, grep-ast, networkx, tiktoken,
platformdirs — and zero numpy/scipy imports.
"""

from __future__ import annotations

from edgemem.slicer.repomap import RepoMap
from edgemem.slicer.tags import Tag, extract_tags
from edgemem.slicer.cache import TagCache

__all__ = ["RepoMap", "Tag", "extract_tags", "TagCache"]
