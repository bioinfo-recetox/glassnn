GlassNN API reference
=====================

GlassNN is a small, transparent neural-network library for teaching. This
site documents its public API: signatures, tensor shapes, the mathematics of
each operation and references. Theory, derivations and tutorials are in the
|book|_.

The package re-exports the most used names: ``glassnn.Tensor`` is
:class:`glassnn.tensor.Tensor`, ``glassnn.no_grad`` is
:class:`glassnn.tensor.no_grad`, ``glassnn.manual_seed`` is
:func:`glassnn.backend.manual_seed`; ``glassnn.nn.Linear`` is
:class:`glassnn.nn.linear.Linear`, and so on for the classes listed in
:mod:`glassnn.nn` and :mod:`glassnn.optim`.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   glassnn.tensor
   glassnn.backend
   glassnn.gradcheck
   glassnn.functional
   glassnn.nn
   glassnn.optim
   glassnn.data
   planned
   bibliography
