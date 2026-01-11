---
title: Distributed Training
order: 3
snippet: >
  ```python
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    
    dist.init_process_group(backend='gloo')
    model = DistributedDataParallel(model)
  ```

summary-home: Scalable distributed training and performance optimization in research and production is enabled by the torch.distributed backend.
featured-home: true

---

* `torch.distributed`
  * == backend /
    * enable
      * scalable distributed training
      * performance optimization | research & production
        * Reason:🧠thanks to
          * native support -- for -- asynchronous execution of collective operations
          * P2P communication / accessible -- from -- Python & C++ 🧠
