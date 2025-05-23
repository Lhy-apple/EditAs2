/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:25:46 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import org.apache.commons.csv.CSVRecord;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CSVRecord_ESTest extends CSVRecord_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, (String) null, 1008L);
      Consumer<String> consumer0 = (Consumer<String>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      cSVRecord0.forEach(consumer0);
      assertEquals(1008L, cSVRecord0.getRecordNumber());
      assertEquals(1, cSVRecord0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String[] stringArray0 = new String[3];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", 0L);
      cSVRecord0.getComment();
      assertEquals("[null, null, null]", cSVRecord0.toString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, (String) null, 1008L);
      String string0 = cSVRecord0.toString();
      assertEquals(1008L, cSVRecord0.getRecordNumber());
      assertEquals("[null]", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "s9+.;", 1166L);
      String[] stringArray1 = cSVRecord0.values();
      assertEquals(1166L, cSVRecord0.getRecordNumber());
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "7w8C&?z8*qqV0S", (-295L));
      // Undeclared exception!
      try { 
        cSVRecord0.get((-1));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -1
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "s9+.;", 1166L);
      long long0 = cSVRecord0.getRecordNumber();
      assertEquals(1166L, long0);
      assertEquals("[null]", cSVRecord0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "?a$71@P>\"smP?/", (-1L));
      int int0 = cSVRecord0.size();
      assertEquals(1, int0);
      assertEquals((-1L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord((String[]) null, hashMap0, (String) null, (-1164L));
      assertEquals("[]", cSVRecord0.toString());
      assertEquals((-1164L), cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", 0L);
      String string0 = cSVRecord0.get("");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String[] stringArray0 = new String[7];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "-(D0B)", 1L);
      // Undeclared exception!
      try { 
        cSVRecord0.get("");
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
      String[] stringArray0 = new String[0];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      Integer integer0 = new Integer((-284));
      hashMap0.put("", integer0);
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "", 0L);
      // Undeclared exception!
      try { 
        cSVRecord0.get("");
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -284
         //
         verifyException("org.apache.commons.csv.CSVRecord", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, (String) null, 1008L);
      boolean boolean0 = cSVRecord0.isConsistent();
      assertEquals(1008L, cSVRecord0.getRecordNumber());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String[] stringArray0 = new String[7];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "No header mapping was specified, the record values can't be accessed by name", (-1L));
      boolean boolean0 = cSVRecord0.isConsistent();
      assertTrue(boolean0);
      assertEquals((-1L), cSVRecord0.getRecordNumber());
      assertEquals("[null, null, null, null, null, null, null]", cSVRecord0.toString());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, (String) null, 1008L);
      assertFalse(cSVRecord0.isConsistent());
      
      Integer integer0 = new Integer(1799);
      hashMap0.put("mv.qCF+4d", integer0);
      boolean boolean0 = cSVRecord0.isConsistent();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String[] stringArray0 = new String[7];
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, (Map<String, Integer>) null, "No header mapping was specified, the record values can't be accessed by name", (-1L));
      boolean boolean0 = cSVRecord0.isMapped("M(yQ$9s;");
      assertEquals((-1L), cSVRecord0.getRecordNumber());
      assertEquals(7, cSVRecord0.size());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      String[] stringArray0 = new String[2];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      Integer integer0 = new Integer((-2089));
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "{5v =yGc3.-3-UH'Zj?", (-835L));
      hashMap0.putIfAbsent("{5v =yGc3.-3-UH'Zj?", integer0);
      boolean boolean0 = cSVRecord0.isSet("{5v =yGc3.-3-UH'Zj?");
      assertEquals((-835L), cSVRecord0.getRecordNumber());
      assertTrue(boolean0);
      assertEquals(2, cSVRecord0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String[] stringArray0 = new String[1];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "s9+.;", 1166L);
      boolean boolean0 = cSVRecord0.isSet("org.apache.commons.csv.CSVRecord");
      assertFalse(boolean0);
      assertEquals("[null]", cSVRecord0.toString());
      assertEquals(1166L, cSVRecord0.getRecordNumber());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String[] stringArray0 = new String[2];
      HashMap<String, Integer> hashMap0 = new HashMap<String, Integer>();
      CSVRecord cSVRecord0 = new CSVRecord(stringArray0, hashMap0, "{5v =yGc3.-3-UH'Zj?", (-835L));
      Integer integer0 = new Integer(2997);
      hashMap0.putIfAbsent("{5v =yGc3.-3-UH'Zj?", integer0);
      boolean boolean0 = cSVRecord0.isSet("{5v =yGc3.-3-UH'Zj?");
      assertEquals("[null, null]", cSVRecord0.toString());
      assertFalse(boolean0);
      assertEquals((-835L), cSVRecord0.getRecordNumber());
  }
}
