Local Neighborhood
==================

The Local Neighborhood algorithm reconstructs a pathway by selecting
all edges adjacent to a user-defined set of relevant nodes (sources,
targets, active nodes, or nodes with a prize).

It is a simple but useful baseline algorithm that does not require
any third-party dependencies. It is mainly used as a reference
implementation for new contributors to SPRAS.

Required inputs
---------------

- ``network``: an edge list in the format ``vertex1|vertex2``.
- ``nodes``: a list of relevant nodes, one per line.

Output format
-------------

The raw output is an edge list in the same ``vertex1|vertex2``
format. SPRAS converts it into the universal pathway format
(``Node1`` ``Node2`` ``Rank`` ``Direction``), with rank ``1`` and
direction ``U`` (undirected) for every edge.

Parameters
----------

This algorithm has no tunable parameters.

Docker image
------------

The Docker image is available at ``reedcompbio/local-neighborhood`` on
Docker Hub. The Dockerfile is in ``docker-wrappers/LocalNeighborhood/``.

References
----------

This algorithm was created as a tutorial example for SPRAS contributors.
See the SPRAS contribution guide for more details. 

