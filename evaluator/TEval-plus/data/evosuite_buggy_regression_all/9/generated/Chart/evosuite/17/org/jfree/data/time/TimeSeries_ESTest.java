/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:32:21 GMT 2023
 */

package org.jfree.data.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import javax.swing.JLayeredPane;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.time.Day;
import org.jfree.data.time.Hour;
import org.jfree.data.time.Minute;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesDataItem;
import org.jfree.data.time.Year;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TimeSeries_ESTest extends TimeSeries_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      Object object0 = timeSeries0.clone();
      boolean boolean0 = timeSeries0.equals(object0);
      assertEquals(1, timeSeries0.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.update(23, (Number) 0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 23, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.getItems();
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
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
  public void test04()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      timeSeries0.getValue((RegularTimePeriod) hour0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.add((RegularTimePeriod) day0, 0.0, false);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) hour0, (Number) 0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are trying to add data where the time period class is org.jfree.data.time.Hour, but the TimeSeries is expecting an instance of org.jfree.data.time.Day.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) hour0, (double) 0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are trying to add data where the time period class is org.jfree.data.time.Hour, but the TimeSeries is expecting an instance of org.jfree.data.time.Day.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      TimeSeries timeSeries1 = new TimeSeries(hour0);
      assertTrue(timeSeries1.equals((Object)timeSeries0));
      
      timeSeries1.setRangeDescription((String) null);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertFalse(timeSeries1.equals((Object)timeSeries0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      
      timeSeries0.setMaximumItemCount(0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      assertEquals(0, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemCount((-2146747451));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'maximum' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      timeSeries0.setMaximumItemCount(0);
      assertEquals(0, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.setMaximumItemAge(0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      RegularTimePeriod regularTimePeriod0 = hour0.previous();
      timeSeries0.addOrUpdate(regularTimePeriod0, (Number) 23);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemAge((-974868L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'periods' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.getDataItem((RegularTimePeriod) hour0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      timeSeries0.getDataItem((RegularTimePeriod) hour0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      timeSeries0.getTimePeriods();
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Integer integer0 = JLayeredPane.POPUP_LAYER;
      TimeSeries timeSeries0 = new TimeSeries(integer0);
      Minute minute0 = new Minute();
      Day day0 = minute0.getDay();
      timeSeries0.add((RegularTimePeriod) day0, (double) 0, true);
      timeSeries0.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TimeSeries timeSeries0 = new TimeSeries("Value");
      MockDate mockDate0 = new MockDate();
      Minute minute0 = new Minute(mockDate0);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (double) 0);
      Class<TimeSeries> class0 = TimeSeries.class;
      TimeSeries timeSeries1 = new TimeSeries("Value", "RkxGs*ul;", "sl^?@Z_c}*E[81", class0);
      timeSeries1.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.getDataItem((RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.getValue((RegularTimePeriod) hour0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
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
  public void test22()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.add((RegularTimePeriod) day0, (double) 23, true);
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) day0, (Number) 23);
      // Undeclared exception!
      try { 
        timeSeries0.add(timeSeriesDataItem0, true);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are attempting to add an observation for the time period 14-February-2014 but the series already contains an observation for that time period. Duplicates are not permitted.  Try using the addOrUpdate() method.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.add((RegularTimePeriod) day0, (double) 23, true);
      RegularTimePeriod regularTimePeriod0 = day0.next();
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem(regularTimePeriod0, (Number) 23);
      timeSeries0.add(timeSeriesDataItem0, true);
      assertEquals(2, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      TimeSeries timeSeries0 = new TimeSeries(mockDate0);
      Minute minute0 = new Minute();
      Day day0 = minute0.getDay();
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      timeSeries0.add((RegularTimePeriod) day0, (double) 23, true);
      timeSeries0.add(regularTimePeriod0, 2902.53187, true);
      assertEquals(2, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Hour hour0 = new Hour();
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      
      timeSeries0.setMaximumItemCount(0);
      timeSeries0.add((RegularTimePeriod) day0, (double) 0, true);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.update((RegularTimePeriod) hour0, (Number) 23);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // TimeSeries.update(TimePeriod, Number):  period does not exist.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.update((RegularTimePeriod) hour0, (Number) 23);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      // Undeclared exception!
      try { 
        timeSeries0.addAndOrUpdate(timeSeries0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are trying to add data where the time period class is org.jfree.data.time.Hour, but the TimeSeries is expecting an instance of org.jfree.data.time.Day.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      TimeSeries timeSeries1 = new TimeSeries(hour0.LAST_HOUR_IN_DAY);
      timeSeries1.addAndOrUpdate(timeSeries0);
      assertEquals(1, timeSeries1.getItemCount());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.addOrUpdate((RegularTimePeriod) null, (Number) 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      RegularTimePeriod regularTimePeriod0 = hour0.previous();
      timeSeries0.addOrUpdate(regularTimePeriod0, (Number) 0);
      timeSeries0.setMaximumItemAge(0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.removeAgedItems((long) 23, true);
      timeSeries0.removeAgedItems((long) 0, true);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      timeSeries0.removeAgedItems((long) 0, false);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Hour hour0 = new Hour();
      Class<Minute> class0 = Minute.class;
      TimeSeries timeSeries0 = new TimeSeries(hour0, class0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      
      timeSeries0.setMaximumItemAge(0L);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      timeSeries0.removeAgedItems((-2427L), true);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Hour hour0 = new Hour();
      Class<Minute> class0 = Minute.class;
      TimeSeries timeSeries0 = new TimeSeries(hour0, class0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      
      timeSeries0.setMaximumItemAge(0L);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      timeSeries0.removeAgedItems((long) 0, false);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.clear();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      timeSeries0.clear();
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.delete((RegularTimePeriod) hour0);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      timeSeries0.delete((RegularTimePeriod) hour0);
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.delete(23, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start <= end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((-4083), 23);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start >= 0.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.clone();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start <= end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      TimeSeries timeSeries1 = timeSeries0.createCopy(0, 0);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertNotSame(timeSeries1, timeSeries0);
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) hour0, (RegularTimePeriod) hour0);
      timeSeries1.setDomainDescription("");
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals("", timeSeries1.getDomainDescription());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) null, (RegularTimePeriod) hour0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'start' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) hour0, (RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'end' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      Day day0 = new Day();
      RegularTimePeriod regularTimePeriod0 = day0.next();
      // Undeclared exception!
      try { 
        timeSeries0.createCopy(regularTimePeriod0, (RegularTimePeriod) day0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start on or before end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      timeSeries0.createCopy((RegularTimePeriod) hour0, (RegularTimePeriod) hour0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      RegularTimePeriod regularTimePeriod0 = hour0.previous();
      timeSeries0.createCopy(regularTimePeriod0, (RegularTimePeriod) hour0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      boolean boolean0 = timeSeries0.equals(timeSeries0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      boolean boolean0 = timeSeries0.equals(hour0);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertFalse(boolean0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      Minute minute0 = new Minute(0, hour0);
      TimeSeries timeSeries1 = new TimeSeries(minute0);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertFalse(boolean0);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      
      timeSeries0.setMaximumItemAge(0);
      TimeSeries timeSeries1 = new TimeSeries(hour0);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      TimeSeries timeSeries1 = new TimeSeries(hour0);
      timeSeries1.setMaximumItemCount(23);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertEquals(23, timeSeries1.getMaximumItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      assertFalse(timeSeries1.equals((Object)timeSeries0));
      
      timeSeries1.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertTrue(timeSeries1.equals((Object)timeSeries0));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 23);
      RegularTimePeriod regularTimePeriod0 = hour0.previous();
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      timeSeries1.addOrUpdate(regularTimePeriod0, (Number) 23);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(1, timeSeries1.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Hour hour0 = new Hour();
      Class<Year> class0 = Year.class;
      TimeSeries timeSeries0 = new TimeSeries(hour0, (String) null, (String) null, class0);
      timeSeries0.hashCode();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.hashCode();
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      timeSeries0.hashCode();
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      RegularTimePeriod regularTimePeriod0 = hour0.previous();
      timeSeries0.addOrUpdate(regularTimePeriod0, (Number) 23);
      timeSeries0.hashCode();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Hour hour0 = new Hour();
      TimeSeries timeSeries0 = new TimeSeries(hour0);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour0, (Number) 0);
      RegularTimePeriod regularTimePeriod0 = hour0.previous();
      MockDate mockDate0 = new MockDate(44, 925, 2, 23, 268);
      timeSeries0.addOrUpdate(regularTimePeriod0, (Number) 23);
      Hour hour1 = new Hour(mockDate0, hour0.DEFAULT_TIME_ZONE);
      timeSeries0.addOrUpdate((RegularTimePeriod) hour1, (Number) null);
      timeSeries0.hashCode();
      assertEquals(3, timeSeries0.getItemCount());
  }
}