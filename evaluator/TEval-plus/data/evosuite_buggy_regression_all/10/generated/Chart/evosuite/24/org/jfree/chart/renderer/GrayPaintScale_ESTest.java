/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:21:16 GMT 2023
 */

package org.jfree.chart.renderer;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.Color;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.renderer.GrayPaintScale;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GrayPaintScale_ESTest extends GrayPaintScale_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = new GrayPaintScale();
      Color color0 = (Color)grayPaintScale0.getPaint(0.0);
      assertEquals(1.0, grayPaintScale0.getUpperBound(), 0.01);
      assertEquals(0, color0.getBlue());
      assertEquals(0.0, grayPaintScale0.getLowerBound(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = new GrayPaintScale();
      double double0 = grayPaintScale0.getLowerBound();
      assertEquals(1.0, grayPaintScale0.getUpperBound(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = new GrayPaintScale();
      GrayPaintScale grayPaintScale1 = (GrayPaintScale)grayPaintScale0.clone();
      boolean boolean0 = grayPaintScale0.equals(grayPaintScale1);
      assertEquals(1.0, grayPaintScale1.getUpperBound(), 0.01);
      assertEquals(0.0, grayPaintScale1.getLowerBound(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = new GrayPaintScale((-612.356745735519), 1737.489);
      double double0 = grayPaintScale0.getUpperBound();
      assertEquals(1737.489, double0, 0.01);
      assertEquals((-612.356745735519), grayPaintScale0.getLowerBound(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = null;
      try {
        grayPaintScale0 = new GrayPaintScale(1956.264567, 1956.264567);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires lowerBound < upperBound.
         //
         verifyException("org.jfree.chart.renderer.GrayPaintScale", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = new GrayPaintScale();
      boolean boolean0 = grayPaintScale0.equals(grayPaintScale0);
      assertEquals(1.0, grayPaintScale0.getUpperBound(), 0.01);
      assertEquals(0.0, grayPaintScale0.getLowerBound(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = new GrayPaintScale();
      Object object0 = new Object();
      boolean boolean0 = grayPaintScale0.equals(object0);
      assertFalse(boolean0);
      assertEquals(1.0, grayPaintScale0.getUpperBound(), 0.01);
      assertEquals(0.0, grayPaintScale0.getLowerBound(), 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = new GrayPaintScale();
      GrayPaintScale grayPaintScale1 = new GrayPaintScale((-1040.351), 3413.4180454);
      boolean boolean0 = grayPaintScale0.equals(grayPaintScale1);
      assertEquals((-1040.351), grayPaintScale1.getLowerBound(), 0.01);
      assertFalse(boolean0);
      assertEquals(3413.4180454, grayPaintScale1.getUpperBound(), 0.01);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      GrayPaintScale grayPaintScale0 = new GrayPaintScale((-106.816675811685), 0.0);
      GrayPaintScale grayPaintScale1 = new GrayPaintScale((-106.816675811685), (-1.0));
      boolean boolean0 = grayPaintScale0.equals(grayPaintScale1);
      assertEquals((-106.816675811685), grayPaintScale1.getLowerBound(), 0.01);
      assertEquals((-1.0), grayPaintScale1.getUpperBound(), 0.01);
      assertFalse(boolean0);
      assertFalse(grayPaintScale1.equals((Object)grayPaintScale0));
  }
}