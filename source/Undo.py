# Stores the blobs when user creates, modify or remove blobs in an array of changes (operations)
# Usage is: addBlob when you create a blob, removeBlob when you (guess) remove a blob
# BUT if you want to modify  blob make a copy, remove the original modify the copy and add the copy.
# similar approach when splitting or joining blobs.

class Undo(object):
    def __init__(self):
        self.position = -1
        self.max_undo = 200
        self.operations = []
        self.operation = { 'remove':[], 'add':[], 'class':[], 'newclass':[] }   #current operation

    def addBlob(self, blob):
        self.operation['remove'].append(blob)

    def removeBlob(self, blob):
        # a region removed twice in the same step (e.g. replaced by several pieces) is restored only once
        if not any(b is blob for b in self.operation['add']):
            self.operation['add'].append(blob)

    def setBlobClass(self, blob, class_name):
        self.operation['class'].append((blob, blob.class_name))
        self.operation['newclass'].append((blob, class_name))

    def saveUndo(self):
        """
        Will mark an undo step using the previously added and removed blobs.
        """
        # a region both added and removed in this step (e.g. created and then undone by a tool) is no change
        removed = {id(b) for b in self.operation['add']}
        both = {id(b) for b in self.operation['remove'] if id(b) in removed}
        if both:
            self.operation['remove'] = [b for b in self.operation['remove'] if id(b) not in both]
            self.operation['add'] = [b for b in self.operation['add'] if id(b) not in both]

        if len(self.operation['add']) == 0 and len(self.operation['remove']) == 0 and len(self.operation['class']) == 0:
            # nothing to undo: the pending step starts again, without what was recorded for its regions
            self.operation = { 'remove':[], 'add':[], 'class':[], 'newclass':[] }
            return

        #clip future redo, invalidated by a new change (only by a real change: an empty step keeps the redo)
        self.operations = self.operations[:self.position+1]

        self.operations.append(self.operation)
        self.operation = { 'remove':[], 'add':[], 'class':[], 'newclass':[] }
        if len(self.operations) > self.max_undo:
            self.operations.pop(0)
        self.position = len(self.operations) - 1

    def mergePendingIntoLast(self):
        """
        Merge the pending step into the last saved one, which must be the current undo position (e.g. a region moved
        again with the arrow keys: one undo step for all the moves). The regions the pending step removes were
        added by the last step: they just disappear from it, and its saved matches (those of the original regions)
        are kept.
        """
        last = self.operations[self.position]
        pending = self.operation
        added_by_last = set(id(b) for b in last['remove'])
        removed_now = set(id(b) for b in pending['add'])
        last['remove'] = [b for b in last['remove'] if id(b) not in removed_now] + pending['remove']
        last['add'] = last['add'] + [b for b in pending['add'] if id(b) not in added_by_last]
        last['class'] = last['class'] + pending['class']
        last['newclass'] = last['newclass'] + pending['newclass']
        self.operation = { 'remove':[], 'add':[], 'class':[], 'newclass':[] }

    def undo(self):
        if len(self.operations) == 0:
            return None
        if self.position < 0:
            return None

        # operation = self.undo_operations.pop(-1)
        operation = self.operations[self.position]
        self.position -= 1
        return operation


    def redo(self):
        if self.position >= len(self.operations) - 1:
            return None
        self.position += 1
        operation = self.operations[self.position]
        return operation

    def peekUndo(self):
        """The operation the next undo() returns, without moving."""
        if self.position < 0 or self.position >= len(self.operations):
            return None
        return self.operations[self.position]

    def peekRedo(self):
        """The operation the next redo() returns, without moving."""
        if self.position >= len(self.operations) - 1:
            return None
        return self.operations[self.position + 1]
