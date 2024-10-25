/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:24:50 GMT 2023
 */

package org.jfree.chart.plot;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.renderer.DefaultPolarItemRenderer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ValueMarker_ESTest extends ValueMarker_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(5.0E-7);
      ValueMarker valueMarker1 = (ValueMarker)valueMarker0.clone();
      boolean boolean0 = valueMarker0.equals(valueMarker1);
      assertTrue(boolean0);
      assertEquals(5.0E-7, valueMarker1.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(2.0);
      double double0 = valueMarker0.getValue();
      assertEquals(2.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(0.16557916939891595);
      valueMarker0.setValue(0.16557916939891595);
      assertEquals(0.16557916939891595, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      DefaultPolarItemRenderer defaultPolarItemRenderer0 = new DefaultPolarItemRenderer();
      ValueMarker valueMarker0 = new ValueMarker((double) defaultPolarItemRenderer0.ZERO, defaultPolarItemRenderer0.DEFAULT_VALUE_LABEL_PAINT, defaultPolarItemRenderer0.DEFAULT_STROKE);
      boolean boolean0 = valueMarker0.equals(defaultPolarItemRenderer0.DEFAULT_VALUE_LABEL_PAINT);
      assertFalse(boolean0);
      assertEquals(0.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(5.0E-7);
      boolean boolean0 = valueMarker0.equals(valueMarker0);
      assertEquals(5.0E-7, valueMarker0.getValue(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(Double.NEGATIVE_INFINITY);
      ValueMarker valueMarker1 = new ValueMarker((-550.62));
      boolean boolean0 = valueMarker0.equals(valueMarker1);
      assertEquals((-550.62), valueMarker1.getValue(), 0.01);
      assertFalse(boolean0);
      assertFalse(valueMarker1.equals((Object)valueMarker0));
  }
}
