# 🌱 Developing guide

## 𓆱 Branching
See [here](https://nvie.com/posts/a-successful-git-branching-model/) the approach to creating branches that is used in the repository.

## 💎 Naming commits
+ `add` means adding functionality
+ `remove` means removing functionality
+ `change` means normal changing
+ `bug` means fixing bug
+ `arch` means changing architecture (usually to be compatible with langchain or other open-source projects)

## Style hints

+ Try to use naming semantic only **once**, change the following code:

| Deprecated          | Recommended         |
|---------------------|---------------------|
| `node_amendment`    | `node`        |
| `children_amendment`    | `children`        |

```python
class NodeAmendmentPropagation(BaseModel):
    """Propagation of amendment among children"""

    node_amendment: DataType
    """Node amendment"""

    children_amendment: Dict[EdgeType, DataType]
    """Children amendment mapping"""
```

This also applies to file names to some extent

+ It will be good if the hyperlinks move to **single** puku-core page. If so, try to mirror *puku -> puku-core* according to this principle, minimizing the number of files globally.

+ Try to group the code in puku files with the same semantics, which is **not** present in puku-core.