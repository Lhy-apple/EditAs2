/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:33:32 GMT 2023
 */

package org.apache.commons.math3.geometry.euclidean.threed;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.apache.commons.math3.geometry.euclidean.oned.IntervalsSet;
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
      Vector3D vector3D0 = new Vector3D(565.97902, 565.97902, 2391.701702);
      SubLine subLine0 = null;
      try {
        subLine0 = new SubLine(vector3D0, (Vector3D) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.geometry.euclidean.threed.Line", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(565.97902, 565.97902, 2391.701702);
      Vector3D vector3D1 = new Vector3D((-2.004495121898943), vector3D0);
      Line line0 = new Line(vector3D0, vector3D1);
      Segment segment0 = new Segment(vector3D1, vector3D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      List<Segment> list0 = subLine0.getSegments();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(565.97902, 565.97902, 2391.701702);
      Vector3D vector3D1 = new Vector3D((-2.004495121898943), vector3D0);
      Line line0 = new Line(vector3D0, vector3D1);
      Segment segment0 = new Segment(vector3D1, vector3D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      Vector3D vector3D2 = subLine0.intersection(subLine0, false);
      assertEquals((-9.788273927731614E-14), vector3D2.getZ(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(565.97902, 565.97902, 2391.702);
      Vector3D vector3D1 = new Vector3D((-1.0), vector3D0);
      Line line0 = new Line(vector3D0, vector3D1);
      Segment segment0 = new Segment(vector3D1, vector3D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      Vector3D vector3D2 = subLine0.intersection(subLine0, true);
      assertNotSame(vector3D1, vector3D2);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(1592.0, 1592.0, 2391.701702);
      Vector3D vector3D1 = Vector3D.PLUS_J;
      Line line0 = new Line(vector3D0, vector3D1);
      Segment segment0 = new Segment(vector3D1, vector3D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      Vector3D vector3D2 = subLine0.intersection(subLine0, true);
      assertNull(vector3D2);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(565.97902, 565.97902, 2391.702);
      Vector3D vector3D1 = new Vector3D((-1.0), vector3D0);
      Line line0 = new Line(vector3D0, vector3D1);
      Segment segment0 = new Segment(vector3D1, vector3D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      IntervalsSet intervalsSet0 = new IntervalsSet(4913.885059489581, (-1.0));
      SubLine subLine1 = new SubLine(line0, intervalsSet0);
      Vector3D vector3D2 = subLine0.intersection(subLine1, true);
      assertNull(vector3D2);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(1592.0, 1592.0, 2391.701702);
      Vector3D vector3D1 = Vector3D.PLUS_J;
      Line line0 = new Line(vector3D0, vector3D1);
      Segment segment0 = new Segment(vector3D1, vector3D0, line0);
      SubLine subLine0 = new SubLine(segment0);
      Vector3D vector3D2 = subLine0.intersection(subLine0, false);
      assertNull(vector3D2);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Vector3D vector3D0 = new Vector3D(565.97902, 565.97902, 2391.701702);
      Vector3D vector3D1 = new Vector3D(0.1, vector3D0, 565.97902, vector3D0, 1.0, vector3D0, 2391.701702, vector3D0);
      Vector3D vector3D2 = new Vector3D((-2.004495121898943), vector3D1);
      Line line0 = new Line(vector3D1, vector3D2);
      Segment segment0 = new Segment(vector3D2, vector3D1, line0);
      SubLine subLine0 = new SubLine(segment0);
      Segment segment1 = new Segment(vector3D0, vector3D1, line0);
      SubLine subLine1 = new SubLine(segment1);
      Vector3D vector3D3 = subLine0.intersection(subLine1, false);
      assertNull(vector3D3);
  }
}
