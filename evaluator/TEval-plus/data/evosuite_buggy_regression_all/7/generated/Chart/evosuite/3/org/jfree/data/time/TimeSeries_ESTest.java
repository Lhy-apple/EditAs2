/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 17:36:42 GMT 2023
 */

package org.jfree.data.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.time.FixedMillisecond;
import org.jfree.data.time.Millisecond;
import org.jfree.data.time.Minute;
import org.jfree.data.time.Month;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesDataItem;
import org.jfree.data.time.Week;
import org.jfree.data.time.Year;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TimeSeries_ESTest extends TimeSeries_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.setDomainDescription("");
      assertEquals("", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0, (String) null, (String) null);
      timeSeries0.add((RegularTimePeriod) week0, (double) 1);
      timeSeries0.addAndOrUpdate(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertTrue(boolean0);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertNotSame(timeSeries1, timeSeries0);
      assertEquals(Double.NaN, timeSeries1.getMaxY(), 0.01);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries1.getMinY(), 0.01);
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals("Value", timeSeries1.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.getItems();
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockDate mockDate0 = new MockDate((-2247), (-2247), (-2247), 0, 0);
      TimeZone timeZone0 = TimeZone.getDefault();
      Year year0 = new Year(mockDate0, timeZone0);
      TimeSeries timeSeries0 = new TimeSeries(year0);
      timeSeries0.getTimePeriodClass();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Week week0 = new Week();
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 1);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add(timeSeriesDataItem0, true);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) week0, 0.5);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are attempting to add an observation for the time period Week 7, 2014 but the series already contains an observation for that time period. Duplicates are not permitted.  Try using the addOrUpdate() method.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, (double) 1);
      timeSeries0.getValue((RegularTimePeriod) week0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Month month0 = new Month();
      TimeSeries timeSeries0 = new TimeSeries(month0);
      double double0 = timeSeries0.getMaxY();
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, double0, 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      double double0 = timeSeries0.getMinY();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Double.NaN, double0, 0.01);
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.delete(53, 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start <= end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, (double) 1);
      timeSeries0.getDataItem((RegularTimePeriod) week0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (double) 0);
      timeSeries0.removeAgedItems((long) 0, true);
      timeSeries0.removeAgedItems((long) 0, true);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.getNextTimePeriod();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemCount((-3610));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'maximum' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      TimeSeries timeSeries0 = new TimeSeries(mockDate0);
      Second second0 = new Second(mockDate0);
      Millisecond millisecond0 = new Millisecond((-706), second0);
      timeSeries0.add((RegularTimePeriod) millisecond0, (Number) 0);
      assertEquals(0.0, timeSeries0.getMinY(), 0.01);
      
      timeSeries0.setMaximumItemCount(0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemAge((-1719L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'periods' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.getDataItem((RegularTimePeriod) week0);
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.getRawDataItem(week0);
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, (double) 1);
      timeSeries0.getRawDataItem(week0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0, "", "");
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (Number) 59);
      timeSeries0.getTimePeriods();
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, (double) 53);
      timeSeries0.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, (double) 53);
      TimeSeries timeSeries1 = new TimeSeries(week0);
      timeSeries1.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(53.0, timeSeries0.getMinY(), 0.01);
      assertEquals(53.0, timeSeries0.getMaxY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.delete((RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.getValue((RegularTimePeriod) week0);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.add((TimeSeriesDataItem) null, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'item' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      Class<RegularTimePeriod> class0 = RegularTimePeriod.class;
      timeSeries0.timePeriodClass = class0;
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) week0, (double) 53);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are trying to add data where the time period class is org.jfree.data.time.Week, but the TimeSeries is expecting an instance of org.jfree.data.time.RegularTimePeriod.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.next();
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 1);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add(timeSeriesDataItem0, true);
      timeSeries0.add(regularTimePeriod0, 0.5);
      timeSeries0.update((RegularTimePeriod) week0, (Number) 1);
      assertEquals(0.5, timeSeries0.getMinY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 1);
      TimeSeries timeSeries0 = new TimeSeries(week0, (String) null, (String) null);
      timeSeries0.setMaximumItemCount(1);
      timeSeries0.add((RegularTimePeriod) week0, (double) 1);
      timeSeries0.add((RegularTimePeriod) week1, (double) 53);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 1);
      timeSeries0.add(timeSeriesDataItem0, false);
      assertEquals(1.0, timeSeries0.getMinY(), 0.01);
      assertEquals(1.0, timeSeries0.getMaxY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0, (String) null, (String) null);
      // Undeclared exception!
      try { 
        timeSeries0.update((RegularTimePeriod) week0, (Number) 1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // There is no existing value for the specified 'period'.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 1);
      TimeSeries timeSeries0 = new TimeSeries(week1);
      timeSeries0.add((RegularTimePeriod) week1, (double) 1);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) null);
      timeSeries0.update(1, (Number) 1);
      assertEquals(1.0, timeSeries0.getMinY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, Double.NaN);
      timeSeries0.update((RegularTimePeriod) week0, (Number) null);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0, (String) null, (String) null);
      timeSeries0.add((RegularTimePeriod) week0, (double) 1);
      TimeSeries timeSeries1 = new TimeSeries(week0);
      timeSeries1.addAndOrUpdate(timeSeries0);
      assertEquals(1, timeSeries1.getItemCount());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.addOrUpdate((TimeSeriesDataItem) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, (double) 53);
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      // Undeclared exception!
      try { 
        timeSeries0.addOrUpdate((RegularTimePeriod) fixedMillisecond0, (Number) 53);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are trying to add data where the time period class is org.jfree.data.time.FixedMillisecond, but the TimeSeries is expecting an instance of org.jfree.data.time.Week.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0, ",-$?", ",-$?");
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) null);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) null);
      assertEquals(",-$?", timeSeries0.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals(",-$?", timeSeries0.getRangeDescription());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.next();
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 1);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add(timeSeriesDataItem0, true);
      timeSeries0.add(regularTimePeriod0, 0.5);
      timeSeries0.addOrUpdate(timeSeriesDataItem0);
      assertEquals(0.5, timeSeries0.getMinY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, Double.NaN);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) 1);
      assertEquals(1.0, timeSeries0.getMinY(), 0.01);
      assertEquals(1.0, timeSeries0.getMaxY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 1);
      TimeSeries timeSeries0 = new TimeSeries(week1);
      timeSeries0.setMaximumItemCount(1);
      timeSeries0.add((RegularTimePeriod) week1, (double) 1);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) 53);
      assertEquals(1, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 1);
      TimeSeries timeSeries0 = new TimeSeries(week1);
      timeSeries0.add((RegularTimePeriod) week1, (double) 1);
      timeSeries0.setMaximumItemAge(53);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) 53);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 1);
      TimeSeries timeSeries0 = new TimeSeries(week0, (String) null, (String) null);
      timeSeries0.add((RegularTimePeriod) week0, (double) 1);
      timeSeries0.addOrUpdate((RegularTimePeriod) week1, (Number) 53);
      timeSeries0.setMaximumItemAge(53);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      Year year0 = new Year(1);
      timeSeries0.addOrUpdate((RegularTimePeriod) year0, (Number) (-9999));
      timeSeries0.removeAgedItems((long) (-9999), true);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.removeAgedItems((long) 0, false);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      Year year0 = new Year(1);
      timeSeries0.addOrUpdate((RegularTimePeriod) year0, (Number) (-9999));
      timeSeries0.setMaximumItemAge(1);
      timeSeries0.removeAgedItems((long) (-9999), true);
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      Year year0 = new Year(1);
      timeSeries0.addOrUpdate((RegularTimePeriod) year0, (Number) (-9999));
      timeSeries0.setMaximumItemAge(1);
      timeSeries0.removeAgedItems(106695L, false);
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.clear();
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, (double) 53);
      timeSeries0.clear();
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.delete((RegularTimePeriod) week0);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Week week0 = new Week();
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 1);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add(timeSeriesDataItem0, true);
      timeSeries0.delete((RegularTimePeriod) week0);
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Minute minute0 = new Minute();
      RegularTimePeriod regularTimePeriod0 = minute0.previous();
      TimeSeries timeSeries0 = new TimeSeries(regularTimePeriod0, "", "G+,!T6F");
      timeSeries0.addOrUpdate(regularTimePeriod0, (Number) 59);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (Number) 0);
      timeSeries0.delete(0, 0, false);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (double) 0);
      timeSeries0.createCopy((RegularTimePeriod) minute0, (RegularTimePeriod) minute0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((-333), 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start >= 0.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy(1, (-1157));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start <= end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = timeSeries0.createCopy(1, 1);
      assertEquals(Double.NaN, timeSeries1.getMinY(), 0.01);
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries1.getMaxY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertNotSame(timeSeries1, timeSeries0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) null, (RegularTimePeriod) minute0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'start' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Year year0 = new Year(0);
      TimeSeries timeSeries0 = new TimeSeries(year0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) year0, (RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'end' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 1);
      TimeSeries timeSeries0 = new TimeSeries(week1);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) week0, (RegularTimePeriod) week1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start on or before end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      TimeZone timeZone0 = TimeZone.getDefault();
      Year year0 = new Year(mockDate0, timeZone0);
      TimeSeries timeSeries0 = new TimeSeries(mockDate0, "", ":Lq%qR)nHrUUhr*O");
      timeSeries0.addOrUpdate((RegularTimePeriod) year0, 0.0);
      RegularTimePeriod regularTimePeriod0 = year0.previous();
      timeSeries0.createCopy(regularTimePeriod0, regularTimePeriod0);
      assertEquals(0.0, timeSeries0.getMinY(), 0.01);
      assertEquals(0.0, timeSeries0.getMaxY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 1);
      TimeSeries timeSeries0 = new TimeSeries(week1);
      timeSeries0.add((RegularTimePeriod) week1, (double) 1);
      timeSeries0.createCopy((RegularTimePeriod) week0, (RegularTimePeriod) week0);
      assertEquals(1.0, timeSeries0.getMaxY(), 0.01);
      assertEquals(1.0, timeSeries0.getMinY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      boolean boolean0 = timeSeries0.equals(timeSeries0);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      boolean boolean0 = timeSeries0.equals(week0);
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertFalse(boolean0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = new TimeSeries(week0, "", "");
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(Double.NaN, timeSeries1.getMaxY(), 0.01);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("", timeSeries1.getRangeDescription());
      assertFalse(boolean0);
      assertEquals("", timeSeries1.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals(Double.NaN, timeSeries1.getMinY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      timeSeries1.setRangeDescription("");
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals("", timeSeries1.getRangeDescription());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      Object object0 = timeSeries0.clone();
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) 53);
      boolean boolean0 = timeSeries0.equals(object0);
      assertEquals(53.0, timeSeries0.getMaxY(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      timeSeries1.setMaximumItemAge(1);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(1L, timeSeries1.getMaximumItemAge());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      timeSeries1.setMaximumItemCount(53);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(53, timeSeries1.getMaximumItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.next();
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 1);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add(timeSeriesDataItem0, true);
      Object object0 = timeSeries0.clone();
      timeSeries0.add(regularTimePeriod0, 0.5);
      Object object1 = timeSeries0.clone();
      boolean boolean0 = object0.equals(object1);
      assertEquals(0.5, timeSeries0.getMinY(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.next();
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 1);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add(timeSeriesDataItem0, true);
      Object object0 = timeSeries0.clone();
      timeSeries0.add(regularTimePeriod0, 0.5);
      timeSeries0.delete((RegularTimePeriod) week0);
      boolean boolean0 = object0.equals(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0, (String) null, (String) null);
      timeSeries0.hashCode();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.hashCode();
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Double.NaN, timeSeries0.getMaxY(), 0.01);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Double.NaN, timeSeries0.getMinY(), 0.01);
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      Year year0 = new Year(1);
      timeSeries0.addOrUpdate((RegularTimePeriod) year0, (Number) (-9999));
      timeSeries0.hashCode();
      assertEquals((-9999.0), timeSeries0.getMinY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.next();
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 1);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add(timeSeriesDataItem0, true);
      timeSeries0.add(regularTimePeriod0, 0.5);
      timeSeries0.hashCode();
      assertEquals(0.5, timeSeries0.getMinY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.next();
      RegularTimePeriod regularTimePeriod1 = regularTimePeriod0.next();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add(regularTimePeriod1, (Number) 53);
      timeSeries0.add((RegularTimePeriod) week0, (double) 1);
      timeSeries0.addOrUpdate(regularTimePeriod0, (Number) 1);
      timeSeries0.hashCode();
      assertEquals(53.0, timeSeries0.getMaxY(), 0.01);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) null);
      timeSeries0.delete((RegularTimePeriod) week0);
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.add((RegularTimePeriod) week0, Double.NaN);
      timeSeries0.delete((RegularTimePeriod) week0);
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 53);
      TimeSeries timeSeries0 = new TimeSeries(week1);
      timeSeries0.add((RegularTimePeriod) week0, Double.NaN);
      timeSeries0.addOrUpdate((RegularTimePeriod) week1, (Number) 1);
      timeSeries0.update((RegularTimePeriod) week1, (Number) 1);
      assertEquals(1.0, timeSeries0.getMaxY(), 0.01);
  }
}
