/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:21:11 GMT 2023
 */

package org.jfree.chart.plot;

import org.junit.Test;
import static org.junit.Assert.*;
import java.awt.Color;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.renderer.DefaultPolarItemRenderer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ValueMarker_ESTest extends ValueMarker_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(14.0);
      ValueMarker valueMarker1 = new ValueMarker((-1.0));
      boolean boolean0 = valueMarker0.equals(valueMarker1);
      assertFalse(boolean0);
      assertFalse(valueMarker1.equals((Object)valueMarker0));
      assertEquals((-1.0), valueMarker1.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(2.0);
      double double0 = valueMarker0.getValue();
      assertEquals(2.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      DefaultPolarItemRenderer defaultPolarItemRenderer0 = new DefaultPolarItemRenderer();
      ValueMarker valueMarker0 = new ValueMarker((double) defaultPolarItemRenderer0.ZERO);
      valueMarker0.setValue((double) defaultPolarItemRenderer0.ZERO);
      assertEquals(0.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      DefaultPolarItemRenderer defaultPolarItemRenderer0 = new DefaultPolarItemRenderer();
      Color color0 = Color.ORANGE;
      ValueMarker valueMarker0 = new ValueMarker((double) defaultPolarItemRenderer0.ZERO, color0, defaultPolarItemRenderer0.DEFAULT_OUTLINE_STROKE);
      assertEquals(0.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker((-200.76316477009));
      boolean boolean0 = valueMarker0.equals(valueMarker0);
      assertTrue(boolean0);
      assertEquals((-200.76316477009), valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      DefaultPolarItemRenderer defaultPolarItemRenderer0 = new DefaultPolarItemRenderer();
      ValueMarker valueMarker0 = new ValueMarker((double) defaultPolarItemRenderer0.ZERO);
      boolean boolean0 = valueMarker0.equals(defaultPolarItemRenderer0);
      assertFalse(boolean0);
      assertEquals(0.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(14.159043230999734);
      ValueMarker valueMarker1 = new ValueMarker(14.159043230999734);
      boolean boolean0 = valueMarker1.equals(valueMarker0);
      assertTrue(boolean0);
      assertEquals(14.159043230999734, valueMarker1.getValue(), 0.01);
  }
}
