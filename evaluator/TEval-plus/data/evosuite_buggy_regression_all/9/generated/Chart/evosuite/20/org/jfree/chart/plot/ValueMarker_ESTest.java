/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:31:11 GMT 2023
 */

package org.jfree.chart.plot;

import org.junit.Test;
import static org.junit.Assert.*;
import java.awt.BasicStroke;
import java.awt.Color;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.axis.PeriodAxisLabelInfo;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.PiePlot;
import org.jfree.chart.plot.ValueMarker;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ValueMarker_ESTest extends ValueMarker_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(0.0);
      ValueMarker valueMarker1 = new ValueMarker((-54.78826300862833));
      boolean boolean0 = valueMarker0.equals(valueMarker1);
      assertFalse(boolean0);
      assertEquals((-54.78826300862833), valueMarker1.getValue(), 0.01);
      assertFalse(valueMarker1.equals((Object)valueMarker0));
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker((-2145.71589001223));
      double double0 = valueMarker0.getValue();
      assertEquals((-2145.71589001223), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(1434.0);
      valueMarker0.setValue(1434.0);
      assertEquals(1434.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Color color0 = (Color)PiePlot.DEFAULT_LABEL_BACKGROUND_PAINT;
      BasicStroke basicStroke0 = (BasicStroke)PeriodAxisLabelInfo.DEFAULT_DIVIDER_STROKE;
      ValueMarker valueMarker0 = new ValueMarker(0.0, color0, basicStroke0);
      assertEquals(0.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(2412.23);
      boolean boolean0 = valueMarker0.equals(valueMarker0);
      assertEquals(2412.23, valueMarker0.getValue(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(1434.0);
      boolean boolean0 = valueMarker0.equals("");
      assertEquals(1434.0, valueMarker0.getValue(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(2412.23);
      IntervalMarker intervalMarker0 = new IntervalMarker(2, (-30.0));
      boolean boolean0 = valueMarker0.equals(intervalMarker0);
      assertFalse(boolean0);
      assertEquals(2412.23, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(0.0);
      ValueMarker valueMarker1 = new ValueMarker(0.0);
      boolean boolean0 = valueMarker1.equals(valueMarker0);
      assertTrue(boolean0);
      assertEquals(0.0, valueMarker1.getValue(), 0.01);
  }
}
