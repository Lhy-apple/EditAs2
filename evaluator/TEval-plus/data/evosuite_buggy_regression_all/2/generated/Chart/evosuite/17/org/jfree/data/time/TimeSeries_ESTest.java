/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:19:49 GMT 2023
 */

package org.jfree.data.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.time.Minute;
import org.jfree.data.time.Month;
import org.jfree.data.time.Quarter;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesDataItem;
import org.jfree.data.time.Week;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TimeSeries_ESTest extends TimeSeries_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries(week0, class0);
      timeSeries0.add((RegularTimePeriod) week0, (Number) 1);
      Object object0 = timeSeries0.clone();
      boolean boolean0 = timeSeries0.equals(object0);
      assertEquals(1, timeSeries0.getItemCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TimeSeries timeSeries0 = new TimeSeries("org.jfree.ata.catFg ry.DefaultCategoryDataset");
      timeSeries0.setDomainDescription("");
      assertEquals("", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.update(53, (Number) 53);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 53, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      timeSeries0.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.getItems();
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Month month0 = new Month();
      TimeSeries timeSeries0 = new TimeSeries(month0);
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
  public void test06()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      timeSeries0.addOrUpdate((RegularTimePeriod) quarter0, (Number) 1);
      timeSeries0.getValue((RegularTimePeriod) quarter0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) week0, (double) 53);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are trying to add data where the time period class is org.jfree.data.time.Week, but the TimeSeries is expecting an instance of org.jfree.data.time.Day.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = new TimeSeries(week0);
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Month month0 = new Month();
      Class<TimeSeriesDataItem> class0 = TimeSeriesDataItem.class;
      TimeSeries timeSeries0 = new TimeSeries(month0, class0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemCount((-146));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'maximum' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Month month0 = new Month();
      Class<TimeSeriesDataItem> class0 = TimeSeriesDataItem.class;
      TimeSeries timeSeries0 = new TimeSeries(month0, class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) month0, (Number) null);
      timeSeries0.setMaximumItemCount(0);
      assertEquals(0, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TimeSeries timeSeries0 = new TimeSeries("orgjfree.ata.catFg ry.DefautCategoryDataset");
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemAge((-280L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'periods' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.getDataItem((RegularTimePeriod) week0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries(week0, class0);
      timeSeries0.add((RegularTimePeriod) week0, (Number) 53);
      TimeSeriesDataItem timeSeriesDataItem0 = timeSeries0.getDataItem((RegularTimePeriod) week0);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertNotNull(timeSeriesDataItem0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      timeSeries0.addOrUpdate((RegularTimePeriod) quarter0, (Number) 1);
      timeSeries0.getTimePeriods();
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      TimeSeries timeSeries1 = new TimeSeries("Requires start >= 0.");
      timeSeries1.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("org.jfree.data.catFg ry.DefaultCategoryDataset", "org.jfree.data.catFg ry.DefaultCategoryDataset", "wH{C6", class0);
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
  public void test17()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      timeSeries0.getValue((RegularTimePeriod) quarter0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.add((TimeSeriesDataItem) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'item' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) week0, (Number) 1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are attempting to add an observation for the time period Week 7, 2014 but the series already contains an observation for that time period. Duplicates are not permitted.  Try using the addOrUpdate() method.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("org.jfree.data.catFg ry.DefaultCategoryDataset", "org.jfree.data.catFg ry.DefaultCategoryDataset", "org.jfree.data.catFg ry.DefaultCategoryDataset", class0);
      timeSeries0.setMaximumItemCount(1);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      timeSeries0.add(regularTimePeriod0, (Number) 53);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries(week0, class0);
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) week0, (Number) 53);
      timeSeries0.add(timeSeriesDataItem0, false);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.update((RegularTimePeriod) week0, (Number) 1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // TimeSeries.update(TimePeriod, Number):  period does not exist.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      timeSeries0.addAndOrUpdate(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Week week0 = new Week();
      Class<Quarter> class0 = Quarter.class;
      TimeSeries timeSeries0 = new TimeSeries("org.jfree.data.catFg ry.DefaultCategoryDatase@", "org.jfree.data.catFg ry.DefaultCategoryDatase@", "org.jfree.data.catFg ry.DefaultCategoryDatase@", class0);
      Class<Minute> class1 = Minute.class;
      TimeSeries timeSeries1 = new TimeSeries(week0, "org.jfree.data.catFg ry.DefaultCategoryDatase@", "org.jfree.data.catFg ry.DefaultCategoryDatase@", class1);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) 53);
      timeSeries1.addAndOrUpdate(timeSeries0);
      assertEquals(1, timeSeries1.getItemCount());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Week week0 = new Week();
      Class<Quarter> class0 = Quarter.class;
      TimeSeries timeSeries0 = new TimeSeries(week0, class0);
      // Undeclared exception!
      try { 
        timeSeries0.addOrUpdate((RegularTimePeriod) null, (double) 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Second second0 = new Second();
      TimeSeries timeSeries0 = new TimeSeries(second0);
      timeSeries0.setMaximumItemCount(0);
      timeSeries0.addOrUpdate((RegularTimePeriod) second0, (Number) 59);
      assertEquals(0, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (-642.0832));
      MockDate mockDate0 = new MockDate(1, 53, 1616);
      Week week1 = new Week(mockDate0);
      timeSeries0.add((RegularTimePeriod) week1, (Number) 53);
      timeSeries0.setMaximumItemAge(1616);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      timeSeries0.add(regularTimePeriod0, (Number) 1);
      RegularTimePeriod regularTimePeriod1 = week0.previous();
      timeSeries0.setMaximumItemAge(1);
      timeSeries0.add(regularTimePeriod1, (Number) 1);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.removeAgedItems((long) 53, true);
      timeSeries0.removeAgedItems((long) 53, true);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      timeSeries0.removeAgedItems(1L, false);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Week week0 = new Week(1, 53);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      timeSeries0.setMaximumItemAge(1L);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.removeAgedItems(1L, false);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Week week0 = new Week(1, 53);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, 188.3182269);
      timeSeries0.setMaximumItemAge(1L);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.removeAgedItems(1L, true);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.clear();
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries(week0, class0);
      timeSeries0.add((RegularTimePeriod) week0, (Number) 53);
      timeSeries0.clear();
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Month month0 = new Month();
      Class<TimeSeriesDataItem> class0 = TimeSeriesDataItem.class;
      TimeSeries timeSeries0 = new TimeSeries(month0, class0);
      timeSeries0.delete((RegularTimePeriod) month0);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) 1);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.delete((RegularTimePeriod) week0);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
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
  public void test38()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((-1220), 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start >= 0.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Month month0 = new Month();
      Class<TimeSeriesDataItem> class0 = TimeSeriesDataItem.class;
      TimeSeries timeSeries0 = new TimeSeries(month0, class0);
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
  public void test40()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries(week0, class0);
      TimeSeries timeSeries1 = timeSeries0.createCopy(1, 53);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertNotSame(timeSeries1, timeSeries0);
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) week0, (RegularTimePeriod) week0);
      assertTrue(timeSeries1.equals((Object)timeSeries0));
      
      timeSeries1.setRangeDescription((String) null);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertFalse(timeSeries1.equals((Object)timeSeries0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TimeSeries timeSeries0 = new TimeSeries("org.jfree.ata.catFg ry.DefaultCategoryDataset");
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) null, (RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'start' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Month month0 = new Month();
      Class<TimeSeriesDataItem> class0 = TimeSeriesDataItem.class;
      TimeSeries timeSeries0 = new TimeSeries(month0, class0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) month0, (RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'end' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      RegularTimePeriod regularTimePeriod0 = week0.previous();
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) week0, regularTimePeriod0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start on or before end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) week0, (RegularTimePeriod) week0);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      timeSeries0.add(regularTimePeriod0, (Number) 1);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) week0, (RegularTimePeriod) week0);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals("Requires start >= 0.", timeSeries1.getRangeDescription());
      assertEquals("Requires start >= 0.", timeSeries1.getDomainDescription());
      assertEquals(0, timeSeries1.getItemCount());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      boolean boolean0 = timeSeries0.equals(timeSeries0);
      assertEquals("Requires start >= 0.", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Requires start >= 0.", timeSeries0.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      boolean boolean0 = timeSeries0.equals(week0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertFalse(boolean0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Week week0 = new Week();
      Class<TimeSeriesDataItem> class0 = TimeSeriesDataItem.class;
      TimeSeries timeSeries0 = new TimeSeries(week0, class0);
      TimeSeries timeSeries1 = new TimeSeries("org.jfree.data.catFg ry.DefaultCategoryDataset");
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertFalse(boolean0);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals("Time", timeSeries1.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      Class<Minute> class0 = Minute.class;
      TimeSeries timeSeries1 = new TimeSeries(week0, ",q3=", ",q3=", class0);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertFalse(boolean0);
      assertEquals(",q3=", timeSeries1.getDomainDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals(",q3=", timeSeries1.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) week0, (RegularTimePeriod) week0);
      timeSeries1.setMaximumItemAge(1000L);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertEquals(1000L, timeSeries1.getMaximumItemAge());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Week week0 = new Week();
      TimeSeries timeSeries0 = new TimeSeries(week0);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) week0, (RegularTimePeriod) week0);
      timeSeries1.setMaximumItemCount(1);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(1, timeSeries1.getMaximumItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("IyN=", "IyN=", "IyN=", class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) week0, (RegularTimePeriod) week0);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries1.update((RegularTimePeriod) week0, (Number) 53);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Month month0 = new Month();
      Class<Quarter> class0 = Quarter.class;
      TimeSeries timeSeries0 = new TimeSeries(month0, (String) null, (String) null, class0);
      timeSeries0.hashCode();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      timeSeries0.hashCode();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Requires start >= 0.", timeSeries0.getDomainDescription());
      assertEquals("Requires start >= 0.", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "Requires start >= 0.", "Requires start >= 0.", class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (double) 53);
      timeSeries0.hashCode();
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Week week0 = new Week();
      Class<Week> class0 = Week.class;
      TimeSeries timeSeries0 = new TimeSeries("Requires start >= 0.", "|R;}w.c]A<EZD_", "2'W1f{B59 uK2%y8?0e", class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, 1.5);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      timeSeries0.add(regularTimePeriod0, (Number) 53);
      RegularTimePeriod regularTimePeriod1 = week0.previous();
      timeSeries0.add(regularTimePeriod1, (Number) 53);
      timeSeries0.hashCode();
      assertEquals(3, timeSeries0.getItemCount());
  }
}