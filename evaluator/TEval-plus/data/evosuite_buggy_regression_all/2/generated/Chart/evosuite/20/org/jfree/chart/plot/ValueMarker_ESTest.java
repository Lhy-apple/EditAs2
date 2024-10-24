/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:19:44 GMT 2023
 */

package org.jfree.chart.plot;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.axis.CategoryAxis3D;
import org.jfree.chart.plot.CombinedDomainCategoryPlot;
import org.jfree.chart.plot.ValueMarker;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ValueMarker_ESTest extends ValueMarker_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(1022.3754899890383);
      ValueMarker valueMarker1 = new ValueMarker(969.451086);
      boolean boolean0 = valueMarker1.equals(valueMarker0);
      assertFalse(boolean0);
      assertFalse(valueMarker0.equals((Object)valueMarker1));
      assertEquals(969.451086, valueMarker1.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(1638.9850629607097);
      double double0 = valueMarker0.getValue();
      assertEquals(1638.9850629607097, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(0.0);
      valueMarker0.setValue(0.0);
      assertEquals(0.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      CombinedDomainCategoryPlot combinedDomainCategoryPlot0 = new CombinedDomainCategoryPlot();
      ValueMarker valueMarker0 = new ValueMarker(10);
      boolean boolean0 = valueMarker0.equals(combinedDomainCategoryPlot0);
      assertFalse(boolean0);
      assertEquals(10.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      ValueMarker valueMarker0 = new ValueMarker(0.0F, categoryAxis3D0.DEFAULT_AXIS_LABEL_PAINT, categoryAxis3D0.DEFAULT_AXIS_LINE_STROKE);
      assertEquals(0.0, valueMarker0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(1022.3754899890383);
      boolean boolean0 = valueMarker0.equals(valueMarker0);
      assertEquals(1022.3754899890383, valueMarker0.getValue(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ValueMarker valueMarker0 = new ValueMarker(1022.3754899890383);
      ValueMarker valueMarker1 = new ValueMarker(1022.3754899890383);
      boolean boolean0 = valueMarker1.equals(valueMarker0);
      assertEquals(1022.3754899890383, valueMarker1.getValue(), 0.01);
      assertTrue(boolean0);
  }
}
