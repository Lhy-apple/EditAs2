/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:39:38 GMT 2023
 */

package org.apache.commons.math.stat.descriptive;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.stat.descriptive.StatisticalSummary;
import org.apache.commons.math.stat.descriptive.StorelessUnivariateStatistic;
import org.apache.commons.math.stat.descriptive.SummaryStatistics;
import org.apache.commons.math.stat.descriptive.moment.GeometricMean;
import org.apache.commons.math.stat.descriptive.moment.SecondMoment;
import org.apache.commons.math.stat.descriptive.moment.Skewness;
import org.apache.commons.math.stat.descriptive.summary.Product;
import org.apache.commons.math.stat.descriptive.summary.Sum;
import org.apache.commons.math.stat.descriptive.summary.SumOfSquares;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SummaryStatistics_ESTest extends SummaryStatistics_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      SummaryStatistics summaryStatistics1 = new SummaryStatistics(summaryStatistics0);
      boolean boolean0 = summaryStatistics1.equals(summaryStatistics0);
      assertTrue(boolean0);
      assertEquals(0L, summaryStatistics1.getN());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      SecondMoment secondMoment0 = new SecondMoment();
      summaryStatistics0.setMeanImpl(secondMoment0);
      assertEquals(0L, summaryStatistics0.getN());
      
      summaryStatistics0.addValue((-2113.4));
      assertEquals(1L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.hashCode();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.getSecondMoment();
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      String string0 = summaryStatistics0.toString();
      assertEquals("SummaryStatistics:\nn: 0\nmin: NaN\nmax: NaN\nmean: NaN\ngeometric mean: NaN\nvariance: NaN\nsum of squares: 0.0\nstandard deviation: NaN\nsum of logs: 0.0\n", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      Skewness skewness0 = new Skewness();
      summaryStatistics0.setSumLogImpl(skewness0);
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      StorelessUnivariateStatistic storelessUnivariateStatistic0 = summaryStatistics0.getVarianceImpl();
      summaryStatistics0.setGeoMeanImpl(storelessUnivariateStatistic0);
      SummaryStatistics summaryStatistics1 = new SummaryStatistics(summaryStatistics0);
      assertEquals(0L, summaryStatistics1.getN());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.getSumLogImpl();
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      StorelessUnivariateStatistic storelessUnivariateStatistic0 = summaryStatistics0.getSumImpl();
      summaryStatistics0.setMaxImpl(storelessUnivariateStatistic0);
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      StatisticalSummary statisticalSummary0 = summaryStatistics0.getSummary();
      assertEquals(0L, statisticalSummary0.getN());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.getMinImpl();
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.getMeanImpl();
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      StorelessUnivariateStatistic storelessUnivariateStatistic0 = summaryStatistics0.getSumsqImpl();
      summaryStatistics0.setVarianceImpl(storelessUnivariateStatistic0);
      SummaryStatistics summaryStatistics1 = new SummaryStatistics();
      boolean boolean0 = summaryStatistics1.equals(summaryStatistics0);
      assertEquals(0L, summaryStatistics1.getN());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.getPopulationVariance();
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.addValue(Double.NaN);
      StorelessUnivariateStatistic storelessUnivariateStatistic0 = summaryStatistics0.getMaxImpl();
      // Undeclared exception!
      try { 
        summaryStatistics0.setSumsqImpl(storelessUnivariateStatistic0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 1 values have been added before statistic is configured
         //
         verifyException("org.apache.commons.math.stat.descriptive.SummaryStatistics", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      Product product0 = new Product();
      summaryStatistics0.setVarianceImpl(product0);
      summaryStatistics0.addValue((-224.37651));
      assertEquals(1L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      GeometricMean geometricMean0 = summaryStatistics0.geoMean;
      StorelessUnivariateStatistic storelessUnivariateStatistic0 = geometricMean0.getSumLogImpl();
      summaryStatistics0.setGeoMeanImpl(storelessUnivariateStatistic0);
      summaryStatistics0.addValue((-1.0));
      assertEquals(1L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.addValue(2990.2309);
      double double0 = summaryStatistics0.getStandardDeviation();
      assertEquals(1L, summaryStatistics0.getN());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.addValue(2990.2309);
      summaryStatistics0.addValue(0.0);
      double double0 = summaryStatistics0.getStandardDeviation();
      assertEquals(2L, summaryStatistics0.getN());
      assertEquals(2114.4125467035533, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.clear();
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      SummaryStatistics.copy(summaryStatistics0, summaryStatistics0);
      summaryStatistics0.clear();
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      boolean boolean0 = summaryStatistics0.equals(summaryStatistics0);
      assertTrue(boolean0);
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      Sum sum0 = new Sum();
      boolean boolean0 = summaryStatistics0.equals(sum0);
      assertFalse(boolean0);
      assertEquals(0L, summaryStatistics0.getN());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      SummaryStatistics summaryStatistics1 = new SummaryStatistics(summaryStatistics0);
      summaryStatistics0.addValue(2990.2309);
      boolean boolean0 = summaryStatistics0.equals(summaryStatistics1);
      assertEquals(1L, summaryStatistics0.getN());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      SummaryStatistics summaryStatistics1 = new SummaryStatistics(summaryStatistics0);
      summaryStatistics0.addValue((-5.004237248003625E181));
      boolean boolean0 = summaryStatistics0.equals(summaryStatistics1);
      assertEquals(1L, summaryStatistics0.getN());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      SummaryStatistics summaryStatistics1 = new SummaryStatistics();
      assertTrue(summaryStatistics1.equals((Object)summaryStatistics0));
      
      Sum sum0 = new Sum();
      summaryStatistics1.setMeanImpl(sum0);
      boolean boolean0 = summaryStatistics0.equals(summaryStatistics1);
      assertFalse(summaryStatistics1.equals((Object)summaryStatistics0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      SummaryStatistics summaryStatistics1 = new SummaryStatistics(summaryStatistics0);
      assertTrue(summaryStatistics1.equals((Object)summaryStatistics0));
      
      SumOfSquares sumOfSquares0 = summaryStatistics1.sumsq;
      summaryStatistics0.setMinImpl(sumOfSquares0);
      boolean boolean0 = summaryStatistics1.equals(summaryStatistics0);
      assertFalse(summaryStatistics1.equals((Object)summaryStatistics0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      summaryStatistics0.n = (-433L);
      SummaryStatistics summaryStatistics1 = new SummaryStatistics();
      boolean boolean0 = summaryStatistics0.equals(summaryStatistics1);
      assertEquals((-433L), summaryStatistics0.getN());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      StorelessUnivariateStatistic storelessUnivariateStatistic0 = summaryStatistics0.getVarianceImpl();
      summaryStatistics0.setSumImpl(storelessUnivariateStatistic0);
      SummaryStatistics summaryStatistics1 = new SummaryStatistics();
      boolean boolean0 = summaryStatistics0.equals(summaryStatistics1);
      assertEquals(0L, summaryStatistics1.getN());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      StorelessUnivariateStatistic storelessUnivariateStatistic0 = summaryStatistics0.getVarianceImpl();
      SummaryStatistics summaryStatistics1 = new SummaryStatistics(summaryStatistics0);
      assertTrue(summaryStatistics1.equals((Object)summaryStatistics0));
      
      summaryStatistics0.setSumsqImpl(storelessUnivariateStatistic0);
      boolean boolean0 = summaryStatistics0.equals(summaryStatistics1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      Sum sum0 = new Sum();
      summaryStatistics0.setVarianceImpl(sum0);
      SummaryStatistics summaryStatistics1 = summaryStatistics0.copy();
      assertTrue(summaryStatistics1.equals((Object)summaryStatistics0));
      assertEquals(0L, summaryStatistics1.getN());
      assertNotSame(summaryStatistics1, summaryStatistics0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      SummaryStatistics summaryStatistics0 = new SummaryStatistics();
      SecondMoment secondMoment0 = new SecondMoment();
      summaryStatistics0.setMeanImpl(secondMoment0);
      SummaryStatistics summaryStatistics1 = new SummaryStatistics(summaryStatistics0);
      assertEquals(0L, summaryStatistics1.getN());
  }
}
