from .cache import *
import time, copy
import abc
class Cache(object):
    """Base implementation of a cache object"""

    @abc.abstractmethod
    def __init__(self, maxlen, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        maxlen : int
            The maximum number of items the cache can store
        """
        raise NotImplementedError('This method is not implemented')

    @abc.abstractmethod
    def __len__(self):
        """Return the number of items currently stored in the cache

        Returns
        -------
        len : int
            The number of items currently in the cache
        """
        raise NotImplementedError('This method is not implemented')

    @property
    @abc.abstractmethod
    def maxlen(self):
        """Return the maximum number of items the cache can store

        Return
        ------
        maxlen : int
            The maximum number of items the cache can store
        """
        raise NotImplementedError('This method is not implemented')

    @abc.abstractmethod
    def dump(self):
        """Return a dump of all the elements currently in the cache possibly
        sorted according to the eviction policy.

        Returns
        -------
        cache_dump : list
            The list of all items currently stored in the cache
        """
        raise NotImplementedError('This method is not implemented')


    def do(self, op, k, *args, **kwargs):
        """Utility method that performs a specified operation on a given item.

        This method allows to perform one of the different operations on an
        item:
         * GET: Retrieve an item
         * PUT: Insert an item
         * UPDATE: Update the value associated to an item
         * DELETE: Remove an item

        Parameters
        ----------
        op : string
            The operation to execute: GET | PUT | UPDATE | DELETE
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        res : bool
            Boolean value being *True* if the operation succeeded or *False*
            otherwise.
        """
        res = {
            'GET':    self.get,
            'PUT':    self.put,
            'UPDATE': self.put,
            'DELETE': self.remove
                }[op](k, *args, **kwargs)
        return res if res is not None else False


    @abc.abstractmethod
    def has(self, k, *args, **kwargs):
        """Check if an item is in the cache without changing the internal
        state of the caching object.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : bool
            Boolean value being *True* if the requested item is in the cache
            or *False* otherwise
        """
        raise NotImplementedError('This method is not implemented')


    @abc.abstractmethod
    def get(self, k, *args, **kwargs):
        """Retrieves an item from the cache.

        Differently from *has(k)*, calling this method may change the internal
        state of the caching object depending on the specific cache
        implementation.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : bool
            Boolean value being *True* if the requested item is in the cache
            or *False* otherwise
        """
        raise NotImplementedError('This method is not implemented')


    @abc.abstractmethod
    def put(self, k, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache, it will not be inserted
        again but the internal state of the cache object may change.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        raise NotImplementedError('This method is not implemented')


    @abc.abstractmethod
    def remove(self, k, *args, **kwargs):
        """Remove an item from the cache, if present.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        removed : bool
            *True* if the content was in the cache, *False* if it was not.
        """
        raise NotImplementedError('This method is not implemented')


    @abc.abstractmethod
    def clear(self):
        """Empty the cache
        """
        raise NotImplementedError('This method is not implemented')
        
class LRUCache(Cache):
    """Least Recently Used (LRU) cache eviction policy.

    According to this policy, When a new item needs to inserted into the cache,
    it evicts the least recently requested one.
    This eviction policy is efficient for line speed operations because both
    search and replacement tasks can be performed in constant time (*O(1)*).

    This policy has been shown to perform well in the presence of temporal
    locality in the request pattern. However, its performance drops under the
    Independent Reference Model (IRM) assumption (i.e. the probability that an
    item is requested is not dependent on previous requests).
    """

    def __init__(self, maxlen, **kwargs):
        self._cache = LinkedSet()
        self.maxSize = int(maxlen)
        self.currentSize = 0
        if self.maxSize <= 0:
            raise ValueError('maxlen must be positive')

    def __len__(self):
        return len(self._cache)

    @property
    def maxlen(self):
        return self.maxSize

    def dump(self):
        return list(iter(self._cache))


    def position(self, k, *args, **kwargs):
        """Return the current position of an item in the cache. Position *0*
        refers to the head of cache (i.e. most recently used item), while
        position *maxlen - 1* refers to the tail of the cache (i.e. the least
        recently used item).

        This method does not change the internal state of the cache.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        position : int
            The current position of the item in the cache
        """
        if not k in self._cache:
            raise ValueError('The item %s is not in the cache' % str(k))
        return self._cache.index(k)


    def has(self, k):
        return k in self._cache


    def get(self, k):
        # search content over the list
        # if it has it push on top, otherwise return false
        if k not in self._cache:
            return -1
        self._cache.move_to_top(k)
        return self._cache.top


    def set(self, id, size):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache, it will pushed to the
        top of the cache.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        # if content in cache, push it on top, no eviction
        size = int(size)
        if id in self._cache:
            self._cache.move_to_top(id)
            return None
        # if content not in cache append it on top
        self._cache.append_top(id, size)
        self.currentSize += size
        while self.currentSize > self.maxSize:
            removedSize = self._cache.pop_bottom()
            self.currentSize -= removedSize
        

    def remove(self, k):
        if k not in self._cache:
            return False
        self._cache.remove(k)
        return True


    def clear(self):
        self._cache.clear()