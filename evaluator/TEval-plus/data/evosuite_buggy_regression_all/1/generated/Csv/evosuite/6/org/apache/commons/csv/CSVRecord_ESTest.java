/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:27:29 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.HashMap;
import java.util.Iterator;
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
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "org.apache.commons.csv.CSVRecord", 0L);
      Iterator<String> iterator0 = cSVRecord0.iterator();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String[] stringArray0 = new String[3];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, (String) null, 831L);
      cSVRecord0.getComment();
      assertEquals("[null, null, null]", cSVRecord0.toString());
      assertEquals(831L, cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "vv_U", 1L);
      cSVRecord0.toMap();
      assertEquals(1L, cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord((String[]) null, hashMap0, " /c:JOO9/,U9#L# ", 2440L);
      String string0 = cSVRecord0.toString();
      assertEquals(2440L, cSVRecord0.getRecordNumber());
      assertEquals("[]", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String[] stringArray0 = new String[3];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "zNXi`SpW", (-3047L));
      String[] stringArray1 = cSVRecord0.values();
      assertEquals((-3047L), cSVRecord0.getRecordNumber());
      assertEquals(3, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord((String[]) null, hashMap0, "5{1Q8_'C", 1923L);
      // Undeclared exception!
      try { 
        cSVRecord0.get((-2118463596));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2118463596
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "OQue", 1037L);
      long long0 = cSVRecord0.getRecordNumber();
      assertEquals(1037L, long0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "No header mapping was specified, the record values can't be accessed by name", (-983L));
      int int0 = cSVRecord0.size();
      assertEquals(1, int0);
      assertEquals((-983L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "brI!b)En", 0L);
      // Undeclared exception!
      try { 
        cSVRecord0.get("$B&O\"?>qIK");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Mapping for $B&O\"?>qIK not found, expected one of []
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String[] stringArray0 = new String[1];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "WtdG&D YqvG/2", (-2969L));
      // Undeclared exception!
      try { 
        cSVRecord0.get("WtdG&D YqvG/2");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No header mapping was specified, the record values can't be accessed by name
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord((String[]) null, hashMap0, "5{1Q8_'C", 1923L);
      Integer integer0 = new Integer(816);
      hashMap0.put("5{1Q8_'C", integer0);
      // Undeclared exception!
      try { 
        cSVRecord0.get("5{1Q8_'C");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Index for header '5{1Q8_'C' is 816 but CSVRecord only has 0 values!
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String[] stringArray0 = new String[3];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "qY4p)b:(xAq{i9i:", 0L);
      boolean boolean0 = cSVRecord0.isConsistent();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String[] stringArray0 = new String[11];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "WtdG&D YqvG/2", (-2969L));
      boolean boolean0 = cSVRecord0.isConsistent();
      assertEquals((-2969L), cSVRecord0.getRecordNumber());
      assertEquals("[null, null, null, null, null, null, null, null, null, null, null]", cSVRecord0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "qY4p)b:(xAq{i9i:", 0L);
      boolean boolean0 = cSVRecord0.isConsistent();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String[] stringArray0 = new String[1];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "WtdG&D YqvG/2", (-2969L));
      boolean boolean0 = cSVRecord0.isMapped("WtdG&D YqvG/2");
      assertFalse(boolean0);
      assertEquals("[null]", cSVRecord0.toString());
      assertEquals((-2969L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord((String[]) null, hashMap0, "", (-404L));
      boolean boolean0 = cSVRecord0.isSet("");
      assertEquals("[]", cSVRecord0.toString());
      assertEquals((-404L), cSVRecord0.getRecordNumber());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String[] stringArray0 = new String[4];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      Integer integer0 = new Integer((-2098677199));
      hashMap0.put("5*?K*!22x%PuPR", integer0);
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "5*?K*!22x%PuPR", (-2098677199));
      boolean boolean0 = cSVRecord0.isSet("5*?K*!22x%PuPR");
      assertTrue(boolean0);
      assertEquals(4, cSVRecord0.size());
      assertEquals((-2098677199L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String[] stringArray0 = new String[2];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, (String) null, 24L);
      Integer integer0 = new Integer(4145);
      hashMap0.put("0\"k`>Iby>cYc#o", integer0);
      boolean boolean0 = cSVRecord0.isSet("0\"k`>Iby>cYc#o");
      assertFalse(boolean0);
      assertEquals(24L, cSVRecord0.getRecordNumber());
      assertEquals("[null, null]", cSVRecord0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      Integer integer0 = new Integer((-3));
      hashMap0.put("vv_U", integer0);
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "vv_U", (-3));
      // Undeclared exception!
      try { 
        cSVRecord0.toMap();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -3
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }
}