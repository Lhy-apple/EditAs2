/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:46:42 GMT 2023
 */

package org.apache.commons.math3.geometry.euclidean.twod;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.List;
import org.apache.commons.math3.geometry.euclidean.oned.Euclidean1D;
import org.apache.commons.math3.geometry.euclidean.twod.Euclidean2D;
import org.apache.commons.math3.geometry.euclidean.twod.Line;
import org.apache.commons.math3.geometry.euclidean.twod.Segment;
import org.apache.commons.math3.geometry.euclidean.twod.SubLine;
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.apache.commons.math3.geometry.partitioning.AbstractSubHyperplane;
import org.apache.commons.math3.geometry.partitioning.Hyperplane;
import org.apache.commons.math3.geometry.partitioning.Side;
import org.apache.commons.math3.geometry.partitioning.SubHyperplane;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SubLine_ESTest extends SubLine_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.ZERO;
      SubLine subLine0 = new SubLine(vector2D0, vector2D0);
      AbstractSubHyperplane<Euclidean2D, Euclidean1D> abstractSubHyperplane0 = subLine0.copySelf();
      assertNotSame(subLine0, abstractSubHyperplane0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Vector2D vector2D0 = new Vector2D((-1092.0), (-1092.0));
      Line line0 = new Line(vector2D0, (-1092.0));
      SubLine subLine0 = line0.wholeHyperplane();
      List<Segment> list0 = subLine0.getSegments();
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.NEGATIVE_INFINITY;
      Line line0 = new Line(vector2D0, vector2D0);
      SubLine subLine0 = line0.wholeHyperplane();
      Vector2D vector2D1 = subLine0.intersection(subLine0, false);
      assertNotSame(vector2D1, vector2D0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Vector2D vector2D0 = new Vector2D((-1092.0), (-1092.0));
      Line line0 = new Line(vector2D0, (-1092.0));
      SubLine subLine0 = line0.wholeHyperplane();
      SubLine subLine1 = new SubLine(vector2D0, vector2D0);
      Vector2D vector2D1 = subLine0.intersection(subLine1, true);
      assertNotSame(vector2D0, vector2D1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.NEGATIVE_INFINITY;
      Line line0 = new Line(vector2D0, vector2D0);
      Segment segment0 = new Segment(vector2D0, vector2D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      Vector2D vector2D1 = subLine0.intersection(subLine0, true);
      assertNull(vector2D1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Vector2D vector2D0 = new Vector2D((-1070.2356198538505), 2234.04913);
      Vector2D vector2D1 = new Vector2D(2234.04913, vector2D0, 0.0, vector2D0);
      SubLine subLine0 = new SubLine(vector2D1, vector2D0);
      Line line0 = new Line(vector2D0, vector2D0);
      Segment segment0 = new Segment(vector2D0, vector2D1, line0);
      SubLine subLine1 = new SubLine(segment0);
      Vector2D vector2D2 = subLine0.intersection(subLine1, true);
      assertNull(vector2D2);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.ZERO;
      Line line0 = new Line(vector2D0, vector2D0);
      Segment segment0 = new Segment(vector2D0, vector2D0, line0);
      Vector2D vector2D1 = Vector2D.POSITIVE_INFINITY;
      SubLine subLine0 = new SubLine(segment0);
      SubLine subLine1 = new SubLine(vector2D1, vector2D0);
      Vector2D vector2D2 = subLine0.intersection(subLine1, false);
      assertNull(vector2D2);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.NEGATIVE_INFINITY;
      Line line0 = new Line(vector2D0, vector2D0);
      SubLine subLine0 = line0.wholeHyperplane();
      Segment segment0 = new Segment(vector2D0, vector2D0, line0);
      SubLine subLine1 = new SubLine(segment0);
      Vector2D vector2D1 = subLine0.intersection(subLine1, false);
      assertNull(vector2D1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.NEGATIVE_INFINITY;
      Line line0 = new Line(vector2D0, vector2D0);
      SubLine subLine0 = line0.wholeHyperplane();
      Side side0 = subLine0.side(line0);
      assertEquals(Side.BOTH, side0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Vector2D vector2D0 = new Vector2D((-1092.0), (-1092.0));
      Line line0 = new Line(vector2D0, (-1092.0));
      SubLine subLine0 = line0.wholeHyperplane();
      Side side0 = subLine0.side(line0);
      assertEquals(Side.HYPER, side0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Vector2D vector2D0 = new Vector2D(1262.34776801, (-0.16666666666666666));
      Vector2D vector2D1 = new Vector2D((-0.16666666666666666), vector2D0);
      Vector2D vector2D2 = new Vector2D((-0.16666666666666666), vector2D1, 1262.34776801, vector2D1, (-0.16666666666666666), vector2D1, 0.008333333333333333, vector2D0);
      Line line0 = new Line(vector2D2, 0.0);
      SubLine subLine0 = line0.wholeHyperplane();
      Line line1 = new Line(vector2D1, 0.0);
      Side side0 = subLine0.side(line1);
      assertEquals(Side.MINUS, side0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Vector2D vector2D0 = new Vector2D(37.657454262637884, 37.657454262637884);
      SubLine subLine0 = new SubLine(vector2D0, vector2D0);
      Vector2D vector2D1 = new Vector2D(37.657454262637884, vector2D0, 280.84, vector2D0, (-854.9204), vector2D0, (-854.9204), vector2D0);
      SubLine subLine1 = new SubLine(vector2D1, vector2D1);
      Hyperplane<Euclidean2D> hyperplane0 = subLine0.getHyperplane();
      Side side0 = subLine1.side(hyperplane0);
      assertEquals(Side.PLUS, side0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.ZERO;
      SubLine subLine0 = new SubLine(vector2D0, vector2D0);
      Line line0 = new Line(vector2D0, 0.5721317290175709);
      Side side0 = subLine0.side(line0);
      assertEquals(Side.PLUS, side0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Vector2D vector2D0 = new Vector2D((-1092.0), (-1092.0));
      Line line0 = new Line(vector2D0, (-1092.0));
      SubLine subLine0 = new SubLine(vector2D0, vector2D0);
      SubHyperplane.SplitSubHyperplane<Euclidean2D> subHyperplane_SplitSubHyperplane0 = subLine0.split(line0);
      assertNotNull(subHyperplane_SplitSubHyperplane0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.ZERO;
      SubLine subLine0 = new SubLine(vector2D0, vector2D0);
      Hyperplane<Euclidean2D> hyperplane0 = subLine0.getHyperplane();
      SubHyperplane.SplitSubHyperplane<Euclidean2D> subHyperplane_SplitSubHyperplane0 = subLine0.split(hyperplane0);
      assertNotNull(subHyperplane_SplitSubHyperplane0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Vector2D vector2D0 = new Vector2D(0.0, 0.0);
      SubLine subLine0 = new SubLine(vector2D0, vector2D0);
      Vector2D vector2D1 = new Vector2D(0.0, (-552.475));
      Line line0 = new Line(vector2D1, vector2D1);
      SubHyperplane.SplitSubHyperplane<Euclidean2D> subHyperplane_SplitSubHyperplane0 = subLine0.split(line0);
      assertNotNull(subHyperplane_SplitSubHyperplane0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.NEGATIVE_INFINITY;
      Line line0 = new Line(vector2D0, vector2D0);
      SubLine subLine0 = line0.wholeHyperplane();
      SubHyperplane.SplitSubHyperplane<Euclidean2D> subHyperplane_SplitSubHyperplane0 = subLine0.split(line0);
      assertNotNull(subHyperplane_SplitSubHyperplane0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Vector2D vector2D0 = Vector2D.NEGATIVE_INFINITY;
      Line line0 = new Line(vector2D0, vector2D0);
      Segment segment0 = new Segment(vector2D0, vector2D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      SubHyperplane.SplitSubHyperplane<Euclidean2D> subHyperplane_SplitSubHyperplane0 = subLine0.split(line0);
      assertNotNull(subHyperplane_SplitSubHyperplane0);
  }
}