GlassNN API reference
=====================

GlassNN is a small, transparent neural-network library for teaching. This
site documents its public API: signatures, tensor shapes, the mathematics of
each operation and references. Theory, derivations and tutorials are in the
|book|_.

The package re-exports the most used names: ``glassnn.Tensor`` is
:class:`glassnn.tensor.Tensor`, ``glassnn.no_grad`` is
:class:`glassnn.tensor.no_grad`, and ``glassnn.backend`` is the module
:mod:`glassnn.backend`.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   glassnn.tensor
   glassnn.backend
   glassnn.gradcheck
   planned
   bibliography
