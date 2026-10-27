import datetime

from source.Blob import Blob
import numpy as np
from source.Blob import Blob
from source.Mask import intersectMask
import pandas as pd


# fates of a matched group recomputed from the areas; any other value set by hand on a matched pair (e.g. "dead",
# a colony that died in place) is kept
COMPUTED_ACTIONS = ("grow", "shrink", "same", "n/s", "born")


class Correspondences(object):

    def __init__(self, img_source, img_target, correspondences = None):

        self.source = img_source
        self.target = img_target
        self.correspondences = []
        self.dead = []
        self.born = []
        self.threshold = 1.05
        self.data = pd.DataFrame(data = correspondences, columns=['Genet', 'Blob1', 'Blob2', 'Area1', 'Area2', 'Class', 'Action', 'Split\Fuse'])
        self.area_type_shown = False  # True means that the surface area is currently shown
        self.repaired = 0  # rows repaired when the table was loaded (see checkTable)

    def area_in_sq_cm(self, area, is_source):

        if is_source:
            area_sq_cm = area * self.source.pixelSize() * self.source.pixelSize() / 100.0
        else:
            area_sq_cm = area * self.target.pixelSize() * self.target.pixelSize() / 100.0

        return area_sq_cm

    def regionArea(self, blob, is_source):
        """
        Area of a region in cm^2: the surface area if the table currently shows it, otherwise the planar area.
        """
        area = blob.surface_area if self.area_type_shown else blob.area
        return self.area_in_sq_cm(float(area or 0.0), is_source)

    def regionsById(self):
        return ({blob.id: blob for blob in self.source.annotations.seg_blobs},
                {blob.id: blob for blob in self.target.annotations.seg_blobs})

    def fate(self, area1, area2):

        if area2 > area1 * self.threshold:
            return "grow"
        elif area2 < area1 / self.threshold:
            return "shrink"
        else:
            return "same"

    def isGenetInfoAvailable(self):

        if len(self.data.index) < 2:
            return False

        if self.data.loc[1, 'Genet'] >= 0:
            return True
        else:
            return False

    def isSource(self, img):

        if self.source == img:
            return True
        else:
            return False

    def groups(self):
        """
        The groups of regions matched together (a colony, or the pieces of a split or a fusion): for each row the
        group key (None for born and dead rows), and the regions of each group as {key: (source ids, target ids)}.
        """
        ids1 = self.data['Blob1'].astype(int).tolist()
        ids2 = self.data['Blob2'].astype(int).tolist()

        parent = {}

        def root(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        for id1, id2 in zip(ids1, ids2):
            if id1 >= 0 and id2 >= 0:
                a = ('s', id1)
                b = ('t', id2)
                parent.setdefault(a, a)
                parent.setdefault(b, b)
                ra = root(a)
                rb = root(b)
                if ra != rb:
                    parent[rb] = ra

        members = {}
        for k in parent:
            group = members.setdefault(root(k), (set(), set()))
            group[0 if k[0] == 's' else 1].add(k[1])

        keys = [root(('s', id1)) if id1 >= 0 and id2 >= 0 else None for id1, id2 in zip(ids1, ids2)]
        return keys, members

    def refresh(self):
        """
        Recompute from the regions the areas, the classes, the fates and the split/fuse labels: the table only
        decides which regions are matched. A split or a fusion gets the fate of the whole group (total area before
        and after).
        """
        if len(self.data.index) == 0:
            return

        src, tgt = self.regionsById()
        ids1 = self.data['Blob1'].astype(int).tolist()
        ids2 = self.data['Blob2'].astype(int).tolist()
        actions = self.data['Action'].tolist()
        classes = self.data['Class'].tolist()

        area1 = [self.regionArea(src[i], True) if i in src else 0.0 for i in ids1]
        area2 = [self.regionArea(tgt[i], False) if i in tgt else 0.0 for i in ids2]

        keys, members = self.groups()
        totals = {}
        for key, (sources, targets) in members.items():
            total1 = sum(self.regionArea(src[i], True) for i in sources if i in src)
            total2 = sum(self.regionArea(tgt[i], False) for i in targets if i in tgt)
            totals[key] = (total1, total2)

        count1 = {}
        count2 = {}
        for id1, id2 in zip(ids1, ids2):
            if id1 >= 0 and id2 >= 0:
                count1[id1] = count1.get(id1, 0) + 1
                count2[id2] = count2.get(id2, 0) + 1

        split_fuse = []
        for r in range(len(ids1)):
            if keys[r] is not None:
                if actions[r] in COMPUTED_ACTIONS:
                    actions[r] = self.fate(*totals[keys[r]])
                if count2[ids2[r]] > 1:
                    split_fuse.append("fuse")
                elif count1[ids1[r]] > 1:
                    split_fuse.append("split")
                else:
                    split_fuse.append("none")
                blob = src.get(ids1[r])
            else:
                actions[r] = "born" if ids1[r] < 0 else "dead"
                split_fuse.append("none")
                blob = src.get(ids1[r]) if ids1[r] >= 0 else tgt.get(ids2[r])
            if blob is not None:
                classes[r] = blob.class_name

        self.data['Area1'] = area1
        self.data['Area2'] = area2
        self.data['Class'] = classes
        self.data['Action'] = actions
        self.data['Split\Fuse'] = split_fuse

    def computedFate(self, row):
        """
        The fate of a matched row (position) computed from the areas of its group (grow, shrink or same).
        """
        keys, members = self.groups()
        if keys[row] is None:
            return None
        src, tgt = self.regionsById()
        sources, targets = members[keys[row]]
        return self.fate(sum(self.regionArea(src[i], True) for i in sources if i in src),
                         sum(self.regionArea(tgt[i], False) for i in targets if i in tgt))

    def updateAreas(self, use_surface_area=False):

        self.area_type_shown = use_surface_area
        self.refresh()

    def updateBlobId(self, img, blob_id, new_id):
        """
        Update the id of a region
        """

        if self.source == img:
            rows_index = self.data[self.data['Blob1'] == blob_id].index
            self.data.loc[rows_index, 'Blob1'] = new_id
        else:
            rows_index = self.data[self.data['Blob2'] == blob_id].index
            self.data.loc[rows_index, 'Blob2'] = new_id

    def updateBlobArea(self, img, blob_id, new_area, new_surface_area):

        # the areas and the fates are always taken from the regions
        self.refresh()

    def save(self):
        return { "source": self.source.id, "target": self.target.id, "correspondences": self.data.values.tolist() }

    def sort_data(self):

        self.data.sort_values(by=['Action', 'Blob1', 'Blob2'], inplace=True, ignore_index=True)

    def appendRows(self, rows):
        """
        Add rows (lists of the 8 values) in one go.
        """
        if len(rows) == 0:
            return

        new = pd.DataFrame(rows, columns=self.data.columns)
        if len(self.data.index) == 0:
            self.data = new
        else:
            self.data = pd.concat([self.data, new], ignore_index=True)

        self.data = self.data.astype({'Genet': int, 'Blob1': int, 'Blob2': int, 'Area1': float, 'Area2': float})

    def unmatchedRow(self, blob, is_source):
        """
        The row of a region without a match: dead (a region of the source map) or born (of the target map).
        """
        if is_source:
            return [-1, blob.id, -1, self.regionArea(blob, True), 0.0, blob.class_name, "dead", "none"]
        else:
            return [-1, -1, blob.id, 0.0, self.regionArea(blob, False), blob.class_name, "born", "none"]

    def columns(self, is_source):
        """
        The column of the regions of the given map, and the column of their partners.
        """
        return ('Blob1', 'Blob2') if is_source else ('Blob2', 'Blob1')

    def addUnmatched(self, blobs, is_source):
        """
        Add a dead (source map) or born (target map) row for each of the regions that have no row yet.
        """
        column, _ = self.columns(is_source)
        present = set(self.data[column])
        rows = [self.unmatchedRow(blob, is_source) for blob in blobs if blob.id not in present]
        self.appendRows(rows)
        return len(rows) > 0

    def removeRegions(self, ids, is_source):
        """
        Remove the rows of the given regions (of the source map if is_source). The partners left without any row
        become dead or born; the other matches are not touched. It returns the ids of the partners.
        """
        column, other = self.columns(is_source)
        ids = set(int(i) for i in ids)
        mask = self.data[column].isin(ids)
        partners = set(int(p) for p in self.data.loc[mask, other]) - {-1}
        self.data = self.data[~mask].reset_index(drop=True)

        left = set(self.data[other])
        src, tgt = self.regionsById()
        regions = tgt if is_source else src
        orphans = [regions[p] for p in sorted(partners) if p not in left and p in regions]
        self.appendRows([self.unmatchedRow(blob, not is_source) for blob in orphans])
        return sorted(partners)

    def linkRegions(self, blobs, partner_ids, is_source):
        """
        Match each region in blobs (of the source map if is_source) to each of the partners (ids of the regions of
        the other map). The other matches of the partners are kept; the born or dead rows of all of them are removed.
        """
        pairs = []
        for blob in blobs:
            for p in sorted(set(int(p) for p in partner_ids)):
                pairs.append((blob.id, p) if is_source else (p, blob.id))
        self.linkPairs(pairs)

    def linkPairs(self, pairs):
        """
        Match the given pairs of regions [(source id, target id)] in one go. The other matches are kept; the born or
        dead rows of the regions of the pairs are removed.
        """
        if len(pairs) == 0:
            return

        ids1 = set(p[0] for p in pairs)
        ids2 = set(p[1] for p in pairs)
        single = (self.data['Blob1'].isin(ids1) & (self.data['Blob2'] < 0)) | \
                 (self.data['Blob2'].isin(ids2) & (self.data['Blob1'] < 0))
        self.data = self.data[~single].reset_index(drop=True)

        src, tgt = self.regionsById()
        existing = set(zip(self.data['Blob1'], self.data['Blob2']))
        rows = []
        for (id1, id2) in pairs:
            if (id1, id2) not in existing:
                existing.add((id1, id2))
                blob = src.get(id1) or tgt.get(id2)
                rows.append([-1, id1, id2, 0.0, 0.0, blob.class_name if blob is not None else "", "n/s", "none"])
        self.appendRows(rows)

    def replaceRegions(self, removed_ids, blobs, is_source):
        """
        The regions removed_ids were replaced by the regions blobs (a cut, a merge, a split...): the new regions take
        over the direct partners of the removed ones (or become dead/born if there are none). It returns the partners.
        """
        column, other = self.columns(is_source)
        mask = self.data[column].isin(set(int(i) for i in removed_ids))
        partners = set(int(p) for p in self.data.loc[mask, other]) - {-1}
        self.data = self.data[~mask].reset_index(drop=True)

        src, tgt = self.regionsById()
        regions = tgt if is_source else src
        partners = sorted(p for p in partners if p in regions)
        if len(partners) > 0:
            self.linkRegions(blobs, partners, is_source)
        else:
            self.addUnmatched(blobs, is_source)
        return partners

    def checkTable(self):
        """
        Repair the table (older versions of TagLab could leave it inconsistent): rows naming regions that no longer
        exist are dropped (a region left without any row becomes dead or born), as are duplicated rows and born/dead
        rows of regions that are matched. Regions (not Empty) missing from the table are added as dead or born.
        The regions concerned are marked to check. It returns the number of rows fixed.
        """
        src, tgt = self.regionsById()
        fixed = 0

        if len(self.data.index) > 0:
            ids1 = self.data['Blob1'].astype(int)
            ids2 = self.data['Blob2'].astype(int)
            missing1 = (ids1 >= 0) & ~ids1.isin(list(src))
            missing2 = (ids2 >= 0) & ~ids2.isin(list(tgt))
            broken = missing1 | missing2
            pair = (ids1 >= 0) & (ids2 >= 0) & ~broken
            matched1 = set(ids1[pair])
            matched2 = set(ids2[pair])
            redundant = ((ids2 < 0) & ids1.isin(matched1)) | ((ids1 < 0) & ids2.isin(matched2))
            empty = (ids1 < 0) & (ids2 < 0)
            duplicated = self.data.duplicated(subset=['Blob1', 'Blob2'])
            drop = broken | redundant | empty | duplicated
            fixed = int(drop.sum())

            survivors1 = set(ids1[broken & ~missing1 & (ids1 >= 0)])
            survivors2 = set(ids2[broken & ~missing2 & (ids2 >= 0)])
            self.data = self.data[~drop].reset_index(drop=True)
            for i in survivors1 | survivors2:
                if i in survivors1:
                    src[i].correspondence_to_check = True
                if i in survivors2:
                    tgt[i].correspondence_to_check = True

        present1 = set(self.data['Blob1'])
        present2 = set(self.data['Blob2'])
        rows = []
        for blob in src.values():
            if blob.id not in present1 and blob.class_name != "Empty":
                rows.append(self.unmatchedRow(blob, True))
                blob.correspondence_to_check = True
        for blob in tgt.values():
            if blob.id not in present2 and blob.class_name != "Empty":
                rows.append(self.unmatchedRow(blob, False))
                blob.correspondence_to_check = True
        self.appendRows(rows)
        fixed += len(rows)

        if fixed > 0:
            self.refresh()
            self.sort_data()

        return fixed

    def fillTable(self, lst):
        """
        Fill the table from a list of correspondences.
        """

        if not lst:
            return

        if len(lst[0]) == 7:
            # genet information is missing..
            for ll in lst:
                ll.insert(0, -1)

        columns = self.data.columns
        self.data = pd.DataFrame(lst, columns=columns)
        self.data = self.data.astype({'Genet': int, 'Blob1': int, 'Blob2': int, 'Area1': float, 'Area2': float})

        # the action list of older versions offered "gone" for "dead"
        self.data.loc[self.data['Action'] == "gone", 'Action'] = "dead"

        self.repaired = self.checkTable()

        #this is needed to ensure consistency between blob data and correspondences data (WHICH SHOULD NOT BE REPLICATED!!!!)
        self.refresh()

        self.sort_data()

    def addBlob(self, image, blob):

        if self.source == image:
            self.set([blob], [])
        elif self.target == image:
            self.set([], [blob])
        else:
            pass

    def removeBlob(self, image, blob):
        if self.source == image:
            self.set([blob], [])
            self.data = self.data[self.data['Blob1'] != blob.id]
        elif self.target == image:
            self.set([], [blob])
            self.data = self.data[self.data['Blob2'] != blob.id]
        else:
            pass

        self.data.reset_index(drop=True, inplace=True)

    def remove(self, img, blob_id):
        """
        Remove the rows with the given blob id.
        """
        pass

    def sourceBlobsById(self, sourceblobs):

        blobs = []
        for id in sourceblobs:
            blob = self.source.annotations.blobById(id)
            if blob is not None:
                blobs.append(blob)

        return blobs

    def targetBlobsById(self, targetblobs):

        blobs = []
        for id in targetblobs:
            blob = self.target.annotations.blobById(id)
            if blob is not None:
                blobs.append(blob)

        return blobs

    def updateBlob(self, image, old_blob, new_blob):

        # check if the class name has changed
        if old_blob.class_name != new_blob.class_name:
            if self.source == image:
                self.set([new_blob], [])
            elif self.target == image:
                self.set([], [new_blob])
            else:
                pass
            return

        if self.source == image:
            self.data.loc[self.data["Blob1"] == old_blob.id, "Blob1"] = new_blob.id
        elif self.target == image:
            self.data.loc[self.data["Blob2"] == old_blob.id, "Blob2"] = new_blob.id
        else:
            pass
        self.refresh()

    def blobClassChanged(self, img, blob, class_name):

        blob_ids = []

        if self.source == img:
            sourceids, targetids, rows = self.findCluster(blob.id, is_source=True)
            for row in rows:
                self.data.loc[row, "Class"] = class_name
                blob_ids.append(self.data.loc[row, "Blob1"])
                blob_ids.append(self.data.loc[row, "Blob2"])
        elif self.target == img:
            sourceids, targetids, rows = self.findCluster(blob.id, is_source=False)
            for row in rows:
                self.data.loc[row, "Class"] = class_name
                blob_ids.append(self.data.loc[row, "Blob1"])
                blob_ids.append(self.data.loc[row, "Blob2"])
        else:
            return

    def set(self, sourceblobs, targetblobs):
        """
        Match the source regions with the target regions (as one group; one of the two lists may be empty: the
        regions are dead or born). Their previous rows are replaced; former partners left without any row become
        dead or born.
        """
        ids1 = [b.id for b in sourceblobs]
        ids2 = [b.id for b in targetblobs]

        involved = self.data['Blob1'].isin(ids1) | self.data['Blob2'].isin(ids2)
        targetorphaned = set(self.data.loc[self.data['Blob1'].isin(ids1), 'Blob2']) - set(ids2) - {-1}
        sourceorphaned = set(self.data.loc[self.data['Blob2'].isin(ids2), 'Blob1']) - set(ids1) - {-1}
        self.data = self.data[~involved].reset_index(drop=True)

        src, tgt = self.regionsById()
        left1 = set(self.data['Blob1'])
        left2 = set(self.data['Blob2'])
        rows = []
        for i in sorted(targetorphaned):
            if i not in left2 and i in tgt:
                rows.append(self.unmatchedRow(tgt[i], False))
        for i in sorted(sourceorphaned):
            if i not in left1 and i in src:
                rows.append(self.unmatchedRow(src[i], True))

        if len(sourceblobs) == 0:
            rows += [self.unmatchedRow(target, False) for target in targetblobs]
        elif len(targetblobs) == 0:
            rows += [self.unmatchedRow(source, True) for source in sourceblobs]
        else:
            for source in sourceblobs:
                for target in targetblobs:
                    rows.append([-1, source.id, target.id, 0.0, 0.0, source.class_name, "n/s", "none"])

        self.appendRows(rows)
        self.refresh()
        self.sort_data()


    def findCluster(self, blobid, is_source):
        """
        All the regions connected to the given one through matches (in the source and in the target map), and the
        indexes of their rows.
        """
        ids1 = self.data['Blob1'].astype(int).tolist()
        ids2 = self.data['Blob2'].astype(int).tolist()
        index = self.data.index.tolist()

        sourcecluster = set()
        targetcluster = set()
        rows = set()
        if is_source:
            sourcecluster.add(blobid)
        else:
            targetcluster.add(blobid)

        todo = [(is_source, blobid)]
        while todo:
            side, blob_id = todo.pop()
            for r in range(len(ids1)):
                if (ids1[r] if side else ids2[r]) != blob_id:
                    continue
                rows.add(index[r])
                other = ids2[r] if side else ids1[r]
                cluster = targetcluster if side else sourcecluster
                if other >= 0 and other not in cluster:
                    cluster.add(other)
                    todo.append((not side, other))

        return list(sourcecluster), list(targetcluster), list(rows)

    def deleteCluster(self, indexes):
        """
        Delete the given rows (positions); the regions left without any row become dead or born.
        """
        rows = self.data.iloc[indexes]
        dead = set(int(i) for i in rows["Blob1"]) - {-1}
        born = set(int(i) for i in rows["Blob2"]) - {-1}

        # delete rows from the dataframe
        self.data = self.data.drop(self.data.index[indexes]).reset_index(drop=True)

        src, tgt = self.regionsById()
        left1 = set(self.data['Blob1'])
        left2 = set(self.data['Blob2'])
        new_rows = [self.unmatchedRow(src[i], True) for i in sorted(dead) if i not in left1 and i in src]
        new_rows += [self.unmatchedRow(tgt[i], False) for i in sorted(born) if i not in left2 and i in tgt]
        self.appendRows(new_rows)

        self.refresh()
        self.sort_data()

    def deleteRows(self, rows_indexes):

        # delete rows from the dataframe
        self.data.drop(rows_indexes, inplace=True)

        # reindexing
        self.data.reset_index(drop=True, inplace=True)

    def exportTidy(self, filename, colonies):
        """
        Save the table as a CSV file for R or a spreadsheet: one line per row, with both regions described (map,
        date, id, class, planar and surface area in cm^2), the fate, the area change and the time between the two
        maps. For a split or a fusion the change is the one of the whole group. colonies maps (image id, region id)
        to the stable colony name (see Genet.colonyNames). The values do not depend on the area shown in the panel.
        """
        src, tgt = self.regionsById()
        px1 = self.source.pixelSize()
        px2 = self.target.pixelSize()

        def cm2(blob, px, surface=False):
            if blob is None:
                return None
            value = float((blob.surface_area if surface else blob.area) or 0.0)
            if surface and value <= 0.0:
                return None
            return round(value * px * px / 100.0, 2)

        years = None
        try:
            date1 = datetime.datetime.strptime(self.source.acquisition_date, '%Y-%m-%d')
            date2 = datetime.datetime.strptime(self.target.acquisition_date, '%Y-%m-%d')
            years = round((date2 - date1).days / 365.25, 3)
        except (TypeError, ValueError):
            pass

        keys, members = self.groups()
        totals = {}
        for key, (sources, targets) in members.items():
            totals[key] = (sum(cm2(src[i], px1) for i in sources if i in src),
                           sum(cm2(tgt[i], px2) for i in targets if i in tgt))

        lines = []
        for r, row in enumerate(self.data.itertuples(index=False)):
            id1 = int(row[1])
            id2 = int(row[2])
            blob1 = src.get(id1) if id1 >= 0 else None
            blob2 = tgt.get(id2) if id2 >= 0 else None
            if keys[r] is not None:
                total1, total2 = totals[keys[r]]
            else:
                total1 = cm2(blob1, px1) or 0.0
                total2 = cm2(blob2, px2) or 0.0
            change = round(total2 - total1, 2)
            change_pct = round(100.0 * change / total1, 1) if total1 > 0 else None

            colony = None
            if blob1 is not None:
                colony = colonies.get((self.source.id, id1))
            if colony is None and blob2 is not None:
                colony = colonies.get((self.target.id, id2))

            to_check = (blob1 is not None and blob1.correspondence_to_check) or \
                       (blob2 is not None and blob2.correspondence_to_check)

            lines.append([colony, int(row[0]),
                          self.source.name, self.source.acquisition_date, id1 if blob1 is not None else None,
                          blob1.class_name if blob1 is not None else None, cm2(blob1, px1), cm2(blob1, px1, True),
                          self.target.name, self.target.acquisition_date, id2 if blob2 is not None else None,
                          blob2.class_name if blob2 is not None else None, cm2(blob2, px2), cm2(blob2, px2, True),
                          row[6], row[7], change, change_pct, years, "TRUE" if to_check else "FALSE"])

        columns = ['colony', 'genet', 'map1', 'date1', 'id1', 'class1', 'area1_cm2', 'surf_area1_cm2',
                   'map2', 'date2', 'id2', 'class2', 'area2_cm2', 'surf_area2_cm2',
                   'fate', 'split_fuse', 'change_cm2', 'change_pct', 'years', 'to_check']
        table = pd.DataFrame(lines, columns=columns)
        table = table.astype({'id1': 'Int64', 'id2': 'Int64'})
        table.sort_values(by=['id1', 'id2'], inplace=True, na_position='last')
        table.to_csv(filename, index=False)

    def autoMatch(self, blobs1, blobs2):
        """
        The regions of the two maps (in the same pixel grid) are matched when they have the same class (not Empty)
        and they overlap by at least 60% of the smaller one.
        """
        self.correspondences.clear()
        boxes2 = np.array([blob.bbox for blob in blobs2], dtype=np.int64).reshape(-1, 4)
        masks2 = {}
        for blob1 in blobs1:
            if blob1.class_name == 'Empty' or len(blobs2) == 0:
                continue

            # the regions whose bounding box intersects the one of blob1 (bbox is top, left, width, height)
            b = blob1.bbox
            rows = np.minimum(b[0] + b[3], boxes2[:, 0] + boxes2[:, 3]) > np.maximum(b[0], boxes2[:, 0])
            cols = np.minimum(b[1] + b[2], boxes2[:, 1] + boxes2[:, 2]) > np.maximum(b[1], boxes2[:, 1])
            mask1 = None

            for j in np.nonzero(rows & cols)[0]:
                blob2 = blobs2[j]
                if blob2.class_name == blob1.class_name:
                    # the masks are computed only for the regions that intersect
                    if mask1 is None:
                        mask1 = Blob.getMask(blob1)
                        sizeblob1 = np.count_nonzero(mask1)
                    if j not in masks2:
                        mask = Blob.getMask(blob2)
                        masks2[j] = (mask, np.count_nonzero(mask))
                    mask2, sizeblob2 = masks2[j]
                    minblob = min(sizeblob1, sizeblob2)
                    mask, bbox = intersectMask(mask1, blob1.bbox, mask2, blob2.bbox)
                    intersectionArea = np.count_nonzero(mask)

                    if (intersectionArea < (0.6 * minblob)):
                        continue
                    if (sizeblob2 > sizeblob1 * self.threshold):
                        self.correspondences.append([-1, blob1.id, blob2.id, blob1.area, blob2.area, blob1.class_name, 'grow', 'none'])

                    elif (sizeblob2 < sizeblob1 / self.threshold):
                        self.correspondences.append([-1, blob1.id, blob2.id, blob1.area, blob2.area, blob1.class_name, 'shrink', 'none'])

                    else:
                        self.correspondences.append([-1, blob1.id, blob2.id, blob1.area, blob2.area, blob1.class_name, 'same', 'none'])


        # operates on the correspondences found and update them
        self.assignSplit()
        self.assignFuse()

        # fill self.born and self.dead blob lists
        self.assignDead(blobs1)
        self.assignBorn(blobs2)


    def autoMatchM(self, blobs1, blobs2):
        self.correspondences.clear()
        for blob1 in blobs1:
            for blob2 in blobs2:
                # use bb to quickly calculate intersection
                x1 = max(blob1.bbox[0], blob2.bbox[0])
                y1 = max(blob1.bbox[1], blob2.bbox[1])
                x2 = min(blob1.bbox[0] + blob1.bbox[3], blob2.bbox[0] + blob2.bbox[3])
                y2 = min(blob1.bbox[1] + blob1.bbox[2], blob2.bbox[1] + blob2.bbox[2])
                # compute the area of intersection rectangle
                interArea = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))

                if interArea != 0 and blob1.class_name != 'Empty':
                    # this is the get mask function for the outer contours, I put it here using two different if conditions so getMask just runs just on intersections
                    mask1 = Blob.getMask(blob1)
                    sizeblob1 = np.count_nonzero(mask1)
                    mask2 = Blob.getMask(blob2)
                    sizeblob2 = np.count_nonzero(mask2)
                    minblob = min(sizeblob1, sizeblob2)
                    mask, bbox = intersectMask(mask1, blob1.bbox, mask2, blob2.bbox)
                    intersectionArea = np.count_nonzero(mask)

                    class_name = blob1.class_name + "-" + blob2.class_name

                    if (intersectionArea < (0.2 * minblob)):
                        continue

                    if (sizeblob2 > sizeblob1 * self.threshold):
                        self.correspondences.append([-1, blob1.id, blob2.id, blob1.area, blob2.area, class_name, 'grow', 'none'])

                    elif (sizeblob2 < sizeblob1 / self.threshold):
                        self.correspondences.append([-1, blob1.id, blob2.id, blob1.area, blob2.area, class_name, 'shrink', 'none'])

                    else:
                        self.correspondences.append([-1, blob1.id, blob2.id, blob1.area, blob2.area, class_name, 'same', 'none'])


        # operates on the correspondences found and update them
        self.assignSplit()
        self.assignFuse()

        # fill self.born and self.dead blob lists
        self.assignDead(blobs1)
        self.assignBorn(blobs2)

        ##### ANALYZE CASES ######

        # Classes are "Pocillopora" and "Pocillopora_dead"

        # 1) D → D
        # 2) D → ∅      (NO MATCH)
        # 3) D or ∅ → L (newborns)
        # 4) L → D       (dead and disappeared)
        # 5) L → ∅      (NO MATCH)
        # 6) L → PD      (PD=partially dead. I will mark this colony as partially dead and then consider only the amount of new dead surface).
        # 7) PD → PD
        # 8) PD → D
        # 9) ∅ → D (error, or suddenly dead)

        # NOTE: case 7 and case 8 now are not considered (!)

        # ADJUST NAMES
        for corresp in self.correspondences:

            class_name = corresp[5]
            class_name1 = corresp[5].split("-")[0]
            class_name2 = corresp[5].split("-")[1]

            if class_name1 == "Pocillopora_dead" and class_name2 == "Pocillopora":
                class_name = "TO CHECK"   # RESURRECTION (!)  (D -> L)

            if class_name1 == "Pocillopora_dead" and class_name2 == "Pocillopora_dead":
                class_name = "DISCARDED"  # case 1 [D → D (discarded)]

            corresp[5] = class_name

        for born in self.born:
            class_name = born[5]
            if class_name == "Pocillopora":
                born[5] = "newborn"    # case 3 [D or ∅ → L (newborns)]
            if class_name == "Pocillopora_dead":
                born[5] = "TO CHECK"   # case 9 [∅ → D (error, or suddenly dead)]

        for dead in self.dead:
            class_name = dead[5]
            if class_name == "Pocillopora":
                dead[5] = "NO MATCH"   # case 5   [L → ∅ (NO MATCH)]
            if class_name == "Pocillopora_dead":
                dead[5] = "DISCARDED"  # case 2   [D → ∅ (DISCARDED)]

        ##### CREATE TABLE AND SAVE IT

        lines = self.correspondences + self.dead + self.born
        self.data = pd.DataFrame(lines, columns=self.data.columns)
        self.sort_data()
        self.data.to_csv("partially_dead.csv", index=False)


        # CHECK IF LIVE POCILLOPORA REMAINS LIVE, BECAME PARTIALLY DEAD, OR TOTALLY DEAD

        id_set = set()
        for corresp in self.correspondences:
            class_name = corresp[5]
            if class_name.find("-") > 0:
                class_name1 = corresp[5].split("-")[0]
                class_name2 = corresp[5].split("-")[1]
                if class_name1 == "Pocillopora":
                    id = corresp[1]
                    id_set.add(id)

        ids = list(id_set)  # list of ids to check

        area_pocillopora_live = 0.0
        area_pocillopora_dead = 0.0
        area_partially_dead_live = 0.0
        area_partially_dead_dead = 0.0
        area_proportion = 0.0
        live = 0
        partially_dead = 0
        totally_dead = 0
        live_data = []
        partially_dead_data = []
        dead_data = []
        for id in ids:
            matches = self.data[self.data['Blob1'] == id]

            area_live = 0.0
            area_dead = 0.0
            for index, row in matches.iterrows():
                class_name = row['Class']
                class_name1 = class_name.split("-")[0]
                class_name2 = class_name.split("-")[1]
                if class_name2 == "Pocillopora":
                    area_live += row['Area2']
                else:
                    area_dead += row['Area2']

            if area_live > 0.0 and area_dead > 0.0:
                # case 6 [L -> PD]
                partially_dead += 1
                area_partially_dead_live += area_live
                area_partially_dead_dead += area_dead
                area_proportion += ((100.0*area_dead) / (area_dead + area_live))

                txt = ""
                for index, row in matches.iterrows():
                    txt = row['Blob2']
                    txt += ","

                txt = txt[:-1]

                partially_dead_data.append([row['Blob1'], txt, row['Area1'], area_live, area_dead])


            if area_dead < 0.00001:
                # case L -> L
                live += 1
                area_pocillopora_live += area_live

                tag = ""
                if area_live > row['Area1']:
                    tag = "Growth"
                else:
                    tag = "Shrink"

                txt = ""
                for index, row in matches.iterrows():
                    txt = row['Blob2']
                    txt += ","

                txt = txt[:-1]

                live_data.append([row['Blob1'], txt, row['Area1'], area_live, tag])

            if area_live < 0.00001:
                # case L -> D
                totally_dead += 1
                area_pocillopora_dead += area_dead

                txt = ""
                for index, row in matches.iterrows():
                    txt = row['Blob2']
                    txt += ","

                txt = txt[:-1]

                dead_data.append([row['Blob1'], txt, row['Area1'], area_dead])


        area_proportion = area_proportion / partially_dead

        data1 = pd.DataFrame(data = live_data, columns=['Blob1', 'Blob2', 'Area1', 'Area2', 'Growth/Shrink'])
        data2 = pd.DataFrame(data = partially_dead_data, columns=['Blob1', 'Blob2', 'Area1', 'Area2 (live)', 'Area2 (dead)'])
        data3 = pd.DataFrame(data = dead_data, columns=['Blob1', 'Blob2', 'Area1', 'Area2'])

        data1.to_csv("2018-2019-plot16-live.csv", index=False)
        data2.to_csv("2018-2019-plot16-partially_dead.csv", index=False)
        data3.to_csv("2018-2019-plot16-dead.csv", index=False)

        ##### COMPUTE STATISTICS

        disappear = 0
        born = 0
        area_disappear = 0.0
        area_born = 0.0
        discarded = 0
        area_discarded = 0.0
        tocheck = 0
        area_tocheck = 0.0
        total_area1 = 0.0
        total_area2 = 0.0
        for index,row in self.data.iterrows():

            if row['Class'] == "newborn":
                born += 1
                area_born += row['Area2']

            if row['Class'] == "NO MATCH":
                disappear += 1
                area_disappear += row['Area1']

            if row['Class'] == "DISCARDED":
                discarded += 1

                if row['Area2'] < 0.00001:
                    area_discarded += row['Area2']
                else:
                    area_discarded += row['Area2']

            if row['Class'] == "TO CHECK":
                tocheck += 1
                if row['Area2'] < 0.00001:
                    area_tocheck += row['Area2']
                else:
                    area_tocheck += row['Area2']

            if row["Blob2"] < 0:
                total_area1 += row['Area1']
            else:
                total_area1 += row['Area2']

        total_area2 = area_pocillopora_live + area_partially_dead_live + area_partially_dead_dead + area_pocillopora_dead + area_born + area_disappear + area_discarded + area_tocheck

        print("Pocillopora live            (L -> L)         : {:d} ({:.2f} cm^2)".format(live, area_pocillopora_live))
        print("Pocillopora partially dead  (L -> PD)        : {:d} (live {:.2f} cm^2) ; dead {:.2f} cm^2) ; % dead {:.2f}".format(partially_dead, area_partially_dead_live,
                                                                               area_partially_dead_dead, area_proportion))
        print("Pocillopora totally dead    (L -> D)         : {:d} ({:.2f} cm^2)".format(totally_dead, area_pocillopora_dead))
        print("Pocillopora born            (∅ -> L)         : {:d} ({:.2f} cm^2)".format(born, area_born))
        print("")

        print("NO MATCH (live or dead)     (D,L → ∅)        : {:d} ({:.2f} cm^2)".format(disappear, area_disappear))

        # note that a dead coral that disappear is still considered a dead coral, this is reasonable in some situations,
        # like when algae cover it but are not annotated, or in other situations where the coral disappear
        # in general we have to pay attention to this to ensure to have reliable results (!)
        print("DEAD REMAINING DEAD         (D -> D, D -> ∅) : {:d} ({:.2f} cm^2)".format(discarded, area_discarded))

        print("TO CHECK                    (∅ → D, D -> L)  : {:d} ({:.2f} cm^2)".format(tocheck, area_tocheck))
        print("")

        print("Total area                                    : {:.2f} cm^2".format(total_area2))


    def assignSplit(self):

        mylist = []
        for i in range(0, len(self.correspondences)):
            mylist.append(int(self.correspondences[i][1]))
        splitted = sorted(set([i for i in mylist if mylist.count(i) > 1]))

        for i in range(0, len(self.correspondences)):
            if int(self.correspondences[i][1]) in splitted:
                self.correspondences[i][7] = 'split'


    def assignFuse(self):

        mylist = []
        for i in range(0, len(self.correspondences)):
            mylist.append(int(self.correspondences[i][2]))
        fused = sorted(set([i for i in mylist if mylist.count(i) > 1]))

        for i in range(0, len(self.correspondences)):
            if int(self.correspondences[i][2]) in fused:
                self.correspondences[i][7] = 'fuse'


    def assignDead(self, blobs1):

        # """
        # Deads are all the blobs that are in project 1 but don't match with any blobs of project 2
        # """
        all_blobs = []
        existing = []
        missing = []

        for i in range(0, len(blobs1)):
            all_blobs.append(int(blobs1[i].id))

        for j in range(0, len(self.correspondences)):
            existing.append(int(self.correspondences[j][1]))

        missing = [i for i in all_blobs if i not in existing]

        for id in missing:
            index = all_blobs.index(id)
            if blobs1[index].class_name != 'Empty':
                self.dead.append([-1, id, -1,  blobs1[index].area, 0.0, blobs1[index].class_name, 'dead', 'none'])


    def assignBorn(self, blobs2):

        # """
        # Borns are all the blobs that are in project 2 but don't match with any blobs of project 1
        # MAYBE NOW MOVED MIGHT BE EXCHANGED FOR NEW BORN
        # """

        all_blobs = []
        existing = []
        missing = []

        for i in range(0, len(blobs2)):
            all_blobs.append(int(blobs2[i].id))

        for j in range(0, len(self.correspondences)):
            existing.append(int(self.correspondences[j][2]))

        missing = [i for i in all_blobs if i not in existing]

        for id in missing:
            index = all_blobs.index(id)
            if blobs2[index].class_name != 'Empty':
                self.born.append([-1, -1, id, 0.0, blobs2[index].area, blobs2[index].class_name, 'born', 'none'])

