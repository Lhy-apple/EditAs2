/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:26:30 GMT 2023
 */

package org.apache.commons.math3.geometry.euclidean.threed;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.List;
import org.apache.commons.math3.geometry.Vector;
import org.apache.commons.math3.geometry.euclidean.oned.IntervalsSet;
import org.apache.commons.math3.geometry.euclidean.threed.Euclidean3D;
import org.apache.commons.math3.geometry.euclidean.threed.Line;
import org.apache.commons.math3.geometry.euclidean.threed.Segment;
import org.apache.commons.math3.geometry.euclidean.threed.SubLine;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SubLine_ESTest extends SubLine_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Vector3D vector3D1 = Vector3D.crossProduct(vector3D0, vector3D0);
      SubLine subLine0 = new SubLine(vector3D0, vector3D1);
      List<Segment> list0 = subLine0.getSegments();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Vector3D vector3D1 = Vector3D.crossProduct(vector3D0, vector3D0);
      SubLine subLine0 = new SubLine(vector3D0, vector3D1);
      Vector3D vector3D2 = subLine0.intersection(subLine0, false);
      assertNull(vector3D2);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Vector3D vector3D1 = Vector3D.crossProduct(vector3D0, vector3D0);
      SubLine subLine0 = new SubLine(vector3D0, vector3D1);
      Vector3D vector3D2 = subLine0.intersection(subLine0, true);
      assertFalse(vector3D2.isNaN());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D((-1909.46971), (-1909.46971), (-1909.46971));
      Vector3D vector3D1 = Vector3D.crossProduct(vector3D0, vector3D0);
      SubLine subLine0 = new SubLine(vector3D0, vector3D1);
      Vector3D vector3D2 = vector3D1.subtract((-3334.06), (Vector<Euclidean3D>) vector3D0);
      SubLine subLine1 = new SubLine(vector3D2, vector3D0);
      Vector3D vector3D3 = subLine1.intersection(subLine0, true);
      assertNull(vector3D3);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Vector3D vector3D1 = Vector3D.crossProduct(vector3D0, vector3D0);
      SubLine subLine0 = new SubLine(vector3D0, vector3D1);
      Line line0 = new Line(vector3D1, vector3D0);
      IntervalsSet intervalsSet0 = new IntervalsSet(1.0, 0.0);
      SubLine subLine1 = new SubLine(line0, intervalsSet0);
      Vector3D vector3D2 = subLine0.intersection(subLine1, true);
      assertNull(vector3D2);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_K;
      Vector3D vector3D1 = vector3D0.orthogonal();
      Vector3D vector3D2 = Vector3D.crossProduct(vector3D0, vector3D0);
      SubLine subLine0 = new SubLine(vector3D0, vector3D2);
      Line line0 = new Line(vector3D2, vector3D1);
      Segment segment0 = new Segment(vector3D1, vector3D0, line0);
      SubLine subLine1 = new SubLine(segment0);
      Vector3D vector3D3 = subLine1.intersection(subLine0, false);
      assertNull(vector3D3);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Vector3D vector3D1 = vector3D0.orthogonal();
      Vector3D vector3D2 = Vector3D.crossProduct(vector3D0, vector3D0);
      Line line0 = new Line(vector3D2, vector3D1);
      Segment segment0 = new Segment(vector3D1, vector3D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      Vector3D vector3D3 = subLine0.intersection(subLine0, false);
      assertEquals(0.0, vector3D3.getZ(), 0.01);
  }
}
