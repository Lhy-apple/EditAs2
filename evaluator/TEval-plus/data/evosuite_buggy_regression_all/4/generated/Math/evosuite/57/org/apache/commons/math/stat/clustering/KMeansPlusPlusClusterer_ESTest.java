/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:11:52 GMT 2023
 */

package org.apache.commons.math.stat.clustering;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import org.apache.commons.math.stat.clustering.Cluster;
import org.apache.commons.math.stat.clustering.EuclideanIntegerPoint;
import org.apache.commons.math.stat.clustering.KMeansPlusPlusClusterer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockRandom;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class KMeansPlusPlusClusterer_ESTest extends KMeansPlusPlusClusterer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      KMeansPlusPlusClusterer<EuclideanIntegerPoint> kMeansPlusPlusClusterer0 = new KMeansPlusPlusClusterer<EuclideanIntegerPoint>(mockRandom0);
      LinkedList<EuclideanIntegerPoint> linkedList0 = new LinkedList<EuclideanIntegerPoint>();
      int[] intArray0 = new int[1];
      EuclideanIntegerPoint euclideanIntegerPoint0 = new EuclideanIntegerPoint(intArray0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      kMeansPlusPlusClusterer0.cluster(linkedList0, 2, 2);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      KMeansPlusPlusClusterer<EuclideanIntegerPoint> kMeansPlusPlusClusterer0 = new KMeansPlusPlusClusterer<EuclideanIntegerPoint>(mockRandom0);
      LinkedList<EuclideanIntegerPoint> linkedList0 = new LinkedList<EuclideanIntegerPoint>();
      int[] intArray0 = new int[1];
      EuclideanIntegerPoint euclideanIntegerPoint0 = new EuclideanIntegerPoint(intArray0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      List<Cluster<EuclideanIntegerPoint>> list0 = kMeansPlusPlusClusterer0.cluster(linkedList0, (-1540), (-1540));
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      LinkedList<EuclideanIntegerPoint> linkedList0 = new LinkedList<EuclideanIntegerPoint>();
      int[] intArray0 = new int[8];
      EuclideanIntegerPoint euclideanIntegerPoint0 = new EuclideanIntegerPoint(intArray0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      KMeansPlusPlusClusterer.EmptyClusterStrategy kMeansPlusPlusClusterer_EmptyClusterStrategy0 = KMeansPlusPlusClusterer.EmptyClusterStrategy.FARTHEST_POINT;
      KMeansPlusPlusClusterer<EuclideanIntegerPoint> kMeansPlusPlusClusterer0 = new KMeansPlusPlusClusterer<EuclideanIntegerPoint>(mockRandom0, kMeansPlusPlusClusterer_EmptyClusterStrategy0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      List<Cluster<EuclideanIntegerPoint>> list0 = kMeansPlusPlusClusterer0.cluster(linkedList0, 2, 2);
      assertEquals(2, list0.size());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      LinkedList<EuclideanIntegerPoint> linkedList0 = new LinkedList<EuclideanIntegerPoint>();
      int[] intArray0 = new int[8];
      EuclideanIntegerPoint euclideanIntegerPoint0 = new EuclideanIntegerPoint(intArray0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      KMeansPlusPlusClusterer.EmptyClusterStrategy kMeansPlusPlusClusterer_EmptyClusterStrategy0 = KMeansPlusPlusClusterer.EmptyClusterStrategy.LARGEST_POINTS_NUMBER;
      KMeansPlusPlusClusterer<EuclideanIntegerPoint> kMeansPlusPlusClusterer0 = new KMeansPlusPlusClusterer<EuclideanIntegerPoint>(mockRandom0, kMeansPlusPlusClusterer_EmptyClusterStrategy0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      List<Cluster<EuclideanIntegerPoint>> list0 = kMeansPlusPlusClusterer0.cluster(linkedList0, 2, 2);
      assertEquals(2, list0.size());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      LinkedList<EuclideanIntegerPoint> linkedList0 = new LinkedList<EuclideanIntegerPoint>();
      int[] intArray0 = new int[19];
      EuclideanIntegerPoint euclideanIntegerPoint0 = new EuclideanIntegerPoint(intArray0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      KMeansPlusPlusClusterer.EmptyClusterStrategy kMeansPlusPlusClusterer_EmptyClusterStrategy0 = KMeansPlusPlusClusterer.EmptyClusterStrategy.ERROR;
      KMeansPlusPlusClusterer<EuclideanIntegerPoint> kMeansPlusPlusClusterer0 = new KMeansPlusPlusClusterer<EuclideanIntegerPoint>(mockRandom0, kMeansPlusPlusClusterer_EmptyClusterStrategy0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      // Undeclared exception!
      try { 
        kMeansPlusPlusClusterer0.cluster(linkedList0, 2, 2);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // empty cluster in k-means
         //
         verifyException("org.apache.commons.math.stat.clustering.KMeansPlusPlusClusterer", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      LinkedList<EuclideanIntegerPoint> linkedList0 = new LinkedList<EuclideanIntegerPoint>();
      int[] intArray0 = new int[1];
      intArray0[0] = Integer.MAX_VALUE;
      EuclideanIntegerPoint euclideanIntegerPoint0 = new EuclideanIntegerPoint(intArray0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      KMeansPlusPlusClusterer.EmptyClusterStrategy kMeansPlusPlusClusterer_EmptyClusterStrategy0 = KMeansPlusPlusClusterer.EmptyClusterStrategy.FARTHEST_POINT;
      KMeansPlusPlusClusterer<EuclideanIntegerPoint> kMeansPlusPlusClusterer0 = new KMeansPlusPlusClusterer<EuclideanIntegerPoint>(mockRandom0, kMeansPlusPlusClusterer_EmptyClusterStrategy0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      List<Cluster<EuclideanIntegerPoint>> list0 = kMeansPlusPlusClusterer0.cluster(linkedList0, (-7503), 50);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      KMeansPlusPlusClusterer<EuclideanIntegerPoint> kMeansPlusPlusClusterer0 = new KMeansPlusPlusClusterer<EuclideanIntegerPoint>(mockRandom0);
      LinkedList<EuclideanIntegerPoint> linkedList0 = new LinkedList<EuclideanIntegerPoint>();
      int[] intArray0 = new int[1];
      EuclideanIntegerPoint euclideanIntegerPoint0 = new EuclideanIntegerPoint(intArray0);
      linkedList0.offerFirst(euclideanIntegerPoint0);
      // Undeclared exception!
      kMeansPlusPlusClusterer0.cluster(linkedList0, 1646, 1646);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom();
      mockRandom0.nextFloat();
      LinkedList<EuclideanIntegerPoint> linkedList0 = new LinkedList<EuclideanIntegerPoint>();
      int[] intArray0 = new int[4];
      intArray0[0] = (-1745);
      EuclideanIntegerPoint euclideanIntegerPoint0 = new EuclideanIntegerPoint(intArray0);
      linkedList0.add(euclideanIntegerPoint0);
      int[] intArray1 = new int[8];
      EuclideanIntegerPoint euclideanIntegerPoint1 = new EuclideanIntegerPoint(intArray1);
      linkedList0.offerFirst(euclideanIntegerPoint1);
      KMeansPlusPlusClusterer.EmptyClusterStrategy kMeansPlusPlusClusterer_EmptyClusterStrategy0 = KMeansPlusPlusClusterer.EmptyClusterStrategy.LARGEST_POINTS_NUMBER;
      KMeansPlusPlusClusterer<EuclideanIntegerPoint> kMeansPlusPlusClusterer0 = new KMeansPlusPlusClusterer<EuclideanIntegerPoint>(mockRandom0, kMeansPlusPlusClusterer_EmptyClusterStrategy0);
      linkedList0.offerFirst(euclideanIntegerPoint1);
      // Undeclared exception!
      try { 
        kMeansPlusPlusClusterer0.cluster(linkedList0, 2, 2);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}
