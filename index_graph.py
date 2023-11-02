#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/23/2023
# @Author  : Ehwartz
# @Github  : https://github.com/Ehwartz
# @Software: PyCharm
# @File    : index_graph.py
import torch
import numpy as np
import cv2


def cos_dist(src, dst):
    return 1 - torch.sum(src * dst) / (torch.norm(src, p=2) * torch.norm(dst, p=2))


def euc_dist(src, dst):
    return torch.sqrt(torch.sum(torch.square(src - dst), dim=-1))


class IndexNode(object):
    def __init__(self, index):
        self.index = index
        self.neibs = []
        self.neibs_index = []

    def connect(self, other):
        if self.index in other.neibs_index or other.index in self.neibs_index:
            return
        self.neibs.append(other)
        self.neibs_index.append(other.index)
        other.neibs.append(self)
        other.neibs_index.append(self.index)


class IndexGraph(object):
    def __init__(self, data: tuple, dist_func):
        x, y = data
        self.n = x.size(0)
        self.nodes = []
        self.ndim = x.size(1)
        self.radius = torch.zeros(size=[self.n])
        for i in range(self.n):
            self.nodes.append(IndexNode(index=i))
        self.adj_matrix = torch.zeros(size=[self.n, self.n])
        for i in range(self.n):
            for j in range(i + 1, self.n):
                src = x[i]
                dst = x[j]
                dist = dist_func(src, dst)
                self.adj_matrix[i][j] = dist

        self.adj_matrix = self.adj_matrix.transpose(1, 0) + self.adj_matrix
        matrix_max = torch.max(self.adj_matrix) + 1
        indices = torch.arange(self.n)
        self.adj_matrix[indices, indices] = matrix_max
        self.clusters = []

    def cluster(self):
        for i in range(self.n):
            dists = self.adj_matrix[i]
            j = torch.argmin(dists)
            if i == j:
                continue
            self.nodes[i].connect(self.nodes[j])

        clusters = []
        for i in range(self.n):
            flag = 1
            for cluster in clusters:
                if i in cluster:
                    flag = 0
                    continue
            if flag:
                c = []
                bfs = BFS(graph=self, cluster=c)
                bfs.search(i)
                clusters.append(c)
        self.clusters = clusters
        return clusters

    def refine(self):
        matrix_min = torch.min(self.adj_matrix) - 1
        indices = torch.arange(self.n)
        self.adj_matrix[indices, indices] = matrix_min
        for c in self.clusters:
            c_dists = []
            for i in c:
                # dists = self.adj_matrix[i][self.nodes[i].neibs_index]
                dists = self.adj_matrix[i][c]
                c_dists.append(dists)
            c_dists = torch.concat(c_dists)

            max_dist = torch.max(c_dists)
            for i in c:
                self.radius[i] = max_dist

            # mean_dist = torch.mean(c_dists)
            # for i in c:
            #     self.radius[i] = mean_dist

        matrix_max = torch.max(self.adj_matrix) + 1
        indices = torch.arange(self.n)
        self.adj_matrix[indices, indices] = matrix_max

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj_matrix[i][j] < self.radius[i]:
                    self.nodes[i].connect(self.nodes[j])
        return self.cluster()


class BFS(object):
    def __init__(self, graph: IndexGraph, cluster: list):
        self.graph = graph
        self.nodes = graph.nodes
        self.queue = []
        self.n = graph.n
        self.visited = np.zeros(shape=[self.n])
        self.cluster = cluster

    def step(self):
        for neib in self.queue[0].neibs:
            i = neib.index
            if self.visited[i]:
                continue
            self.visited[i] = 1
            self.cluster.append(i)
            self.queue.append(neib)
        self.queue.pop(0)

    def search(self, src):
        self.queue.clear()
        self.cluster.append(src)
        self.visited[src] = 1
        self.queue.append(self.nodes[src])
        while self.queue:
            self.step()
            # print(len(self.queue))


def data_init(centers, n, r):
    dim = centers[0].size(0)
    xs = []

    for center in centers:
        for i in range(n):
            xs.append(center + (2 * torch.rand(size=center.size()) - 1) * r)
    return torch.concat(xs, dim=0).view([-1, dim])


def index_img_show(xs, index):
    img = xs[index].detach().view([28, 28]).numpy()
    cv2.imshow('img', img)
    cv2.waitKey(0)


def cluster_img_show(xs, c):
    print(len(c), c)
    for i in c:
        index_img_show(xs, i)


def clusters_img_show(xs, clusters):
    for c in clusters:
        cluster_img_show(xs, c)


def splice_cluster_images(cluster, xs, max_n, imgsz):
    ret_img = np.zeros(shape=[imgsz[0] * max_n, imgsz[1] * max_n])
    for i, c in enumerate(clusters):
        if i >= max_n:
            return ret_img
        for j, index in enumerate(c):
            if j < max_n:
                img = xs[index].detach().view([imgsz[0], imgsz[1]]).numpy()
                ret_img[i * imgsz[0]:(i + 1) * imgsz[0], j * imgsz[1]:(j + 1) * imgsz[1]] = img

    return ret_img


if __name__ == '__main__':
    from data import MNIST

    dataset = MNIST(root='./datasets', train=True, flatten=True)
    xs, ys, indices = dataset.filter(labels=[0])[:1000]
    graph = IndexGraph(data=(xs, ys), dist_func=euc_dist)

    clusters = graph.cluster()
    for c in clusters:
        print(len(c), c)
        # print((torch.argmax(ys[c], dim=-1)))
    print(len(clusters))
    print('\n\n\n')
    #
    for i in range(3):
        print('iter:  ', i)
        clusters = graph.refine()
        for c in clusters:
            print(len(c), c)
            # print((torch.argmax(ys[c], dim=-1)))
        print(len(clusters))
        print('\n\n\n')

    for k in range(1, 10):
        xs, ys, indices = dataset.filter(labels=[k])[:1000]
        graph = IndexGraph(data=(xs, ys), dist_func=euc_dist)
        clusters = graph.cluster()
        clusters = graph.refine()
        spliced_img = splice_cluster_images(graph.clusters, xs, 40, (28, 28)) * 255
        cv2.imwrite(f'./clusters/2clusters{k}.png', spliced_img)
        cv2.imshow('img', spliced_img)
        cv2.waitKey(0)



    # clusters_img_show(xs, graph.clusters)
