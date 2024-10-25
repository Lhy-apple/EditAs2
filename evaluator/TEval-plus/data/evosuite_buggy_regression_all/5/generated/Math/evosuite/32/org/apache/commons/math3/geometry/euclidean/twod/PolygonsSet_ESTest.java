/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:30:21 GMT 2023
 */

package org.apache.commons.math3.geometry.euclidean.twod;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import org.apache.commons.math3.geometry.euclidean.oned.Euclidean1D;
import org.apache.commons.math3.geometry.euclidean.twod.Euclidean2D;
import org.apache.commons.math3.geometry.euclidean.twod.PolygonsSet;
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.apache.commons.math3.geometry.partitioning.AbstractRegion;
import org.apache.commons.math3.geometry.partitioning.BSPTree;
import org.apache.commons.math3.geometry.partitioning.SubHyperplane;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PolygonsSet_ESTest extends PolygonsSet_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet((-1.0E-10), 0.0, (-3.141592653589793), (-1.0E-10));
      polygonsSet0.computeGeometricalProperties();
      assertFalse(polygonsSet0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet();
      AbstractRegion<Euclidean2D, Euclidean1D> abstractRegion0 = polygonsSet0.copySelf();
      assertNotSame(polygonsSet0, abstractRegion0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<SubHyperplane<Euclidean2D>> linkedList0 = new LinkedList<SubHyperplane<Euclidean2D>>();
      PolygonsSet polygonsSet0 = new PolygonsSet(linkedList0);
      assertFalse(polygonsSet0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet();
      polygonsSet0.computeGeometricalProperties();
      assertFalse(polygonsSet0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Boolean boolean0 = Boolean.FALSE;
      BSPTree<Euclidean2D> bSPTree0 = new BSPTree<Euclidean2D>(boolean0);
      PolygonsSet polygonsSet0 = new PolygonsSet(bSPTree0);
      polygonsSet0.computeGeometricalProperties();
      assertTrue(polygonsSet0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet(1.7976931348623157E308, 0.6876171058980829, 0.6876171058980829, 3.356118100840571E-7);
      polygonsSet0.computeGeometricalProperties();
      assertFalse(polygonsSet0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet();
      polygonsSet0.getVertices();
      Vector2D[][] vector2DArray0 = polygonsSet0.getVertices();
      assertEquals(0, vector2DArray0.length);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet(0.0, 1.0000000000000003E-10, 1157.4873448, 1296.1);
      // Undeclared exception!
      try { 
        polygonsSet0.computeGeometricalProperties();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.apache.commons.math3.geometry.partitioning.BoundaryAttribute cannot be cast to java.lang.Boolean
         //
         verifyException("org.apache.commons.math3.geometry.euclidean.twod.PolygonsSet", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet(894.08228, 1.7976931348623157E308, (-920.1446683), 894.08228);
      polygonsSet0.computeGeometricalProperties();
      assertFalse(polygonsSet0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet(1.7976931348623157E308, (-2746.0), 1.7976931348623157E308, (-2746.0));
      // Undeclared exception!
      try { 
        polygonsSet0.computeGeometricalProperties();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: internal error, please fill a bug report at https://issues.apache.org/jira/browse/MATH
         //
         verifyException("org.apache.commons.math3.geometry.euclidean.twod.PolygonsSet", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PolygonsSet polygonsSet0 = new PolygonsSet(1.7976931348623157E308, 3.356118100840571E-7, 1.7976931348623157E308, 3.356118100840571E-7);
      // Undeclared exception!
      try { 
        polygonsSet0.computeGeometricalProperties();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.geometry.euclidean.twod.Line", e);
      }
  }
}
