/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:29:20 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.csv.CSVRecord;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVRecord_ESTest extends CSVRecord_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", 474L);
      cSVRecord0.iterator();
      assertEquals(474L, cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", 474L);
      cSVRecord0.getComment();
      assertEquals(474L, cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord((String[]) null, hashMap0, "org.apache.commons.csv.CSVRecord", 1464L);
      String string0 = cSVRecord0.toString();
      assertEquals(1464L, cSVRecord0.getRecordNumber());
      assertEquals("[]", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", 474L);
      String[] stringArray1 = cSVRecord0.values();
      assertSame(stringArray1, stringArray0);
      assertEquals(474L, cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", (-2548L));
      // Undeclared exception!
      try { 
        cSVRecord0.get(266);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 266
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", (-2548L));
      long long0 = cSVRecord0.getRecordNumber();
      assertEquals((-2548L), long0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", (-2548L));
      cSVRecord0.size();
      assertEquals((-2548L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", (-2548L));
      cSVRecord0.get("$w");
      assertEquals((-2548L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      String[] stringArray0 = new String[1];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "", 0L);
      // Undeclared exception!
      try { 
        cSVRecord0.get(".97^+ 0W}2");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No header mapping was specified, the record values can't be accessed by name
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      Integer integer0 = new Integer((-1));
      hashMap0.put(",]!0k*hZMN6`EqzUN", integer0);
      CSVRecord cSVRecord0 = new CSVRecord((String[]) null, hashMap0, "J\"6C}|GtghZVEa", 0L);
      // Undeclared exception!
      try { 
        cSVRecord0.get(",]!0k*hZMN6`EqzUN");
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -1
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", (-2548L));
      boolean boolean0 = cSVRecord0.isConsistent();
      assertEquals((-2548L), cSVRecord0.getRecordNumber());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String[] stringArray0 = new String[1];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "fr)m_j)J.4M!*Rf~>%", (-266L));
      boolean boolean0 = cSVRecord0.isConsistent();
      assertEquals("[null]", cSVRecord0.toString());
      assertEquals((-266L), cSVRecord0.getRecordNumber());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String[] stringArray0 = new String[4];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, ")*", (-1L));
      boolean boolean0 = cSVRecord0.isConsistent();
      assertFalse(boolean0);
      assertEquals((-1L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      String[] stringArray0 = new String[1];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "fr)m_j)J.4M!*Rf~>%", (-266L));
      boolean boolean0 = cSVRecord0.isSet("+");
      assertFalse(boolean0);
      assertEquals(1, cSVRecord0.size());
      assertEquals((-266L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", (-2548L));
      boolean boolean0 = cSVRecord0.isSet("No header mapping was specified, the record values can't be accessed by name");
      assertEquals((-2548L), cSVRecord0.getRecordNumber());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      String[] stringArray0 = new String[4];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      Integer integer0 = new Integer((-2));
      hashMap0.put("", integer0);
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, ")*", (-1L));
      boolean boolean0 = cSVRecord0.isSet("");
      assertTrue(boolean0);
      assertEquals((-1L), cSVRecord0.getRecordNumber());
      assertEquals(4, cSVRecord0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      Integer integer0 = new Integer(1846);
      hashMap0.put("", integer0);
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", 474L);
      boolean boolean0 = cSVRecord0.isSet("");
      assertEquals(474L, cSVRecord0.getRecordNumber());
      assertFalse(boolean0);
  }
}