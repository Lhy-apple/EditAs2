/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:52:44 GMT 2023
 */

package com.fasterxml.jackson.core;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonPointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonPointer_ESTest extends JsonPointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      jsonPointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("H7%da\"");
      String string0 = jsonPointer0.getMatchingProperty();
      assertEquals("7%da\"", string0);
      assertEquals("H7%da\"", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
      assertNotNull(string0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.tail();
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      String string0 = jsonPointer0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        JsonPointer.valueOf("09143230167");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid input: JSON Pointer expression must start with '/': \"09143230167\"
         //
         verifyException("com.fasterxml.jackson.core.JsonPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      int int0 = jsonPointer0.getMatchingIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf((String) null);
      assertEquals("", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile("");
      assertTrue(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile("/z\"w$~)x>0 =^");
      assertFalse(jsonPointer0.matches());
      assertEquals("z\"w$~)x>0 =^", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("a:#iM-4ZI1 s.$");
      boolean boolean0 = jsonPointer0.matches();
      assertFalse(boolean0);
      assertEquals(":#iM-4ZI1 s.$", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("a:#iM-4ZI1 s.$", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.matches();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("09143230167");
      boolean boolean0 = jsonPointer0.mayMatchProperty();
      assertEquals("09143230167", jsonPointer0.toString());
      assertTrue(boolean0);
      assertFalse(jsonPointer0.matches());
      assertEquals("9143230167", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("\"");
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertEquals("", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
      assertEquals("\"", jsonPointer0.toString());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("_8");
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertFalse(jsonPointer0.matches());
      assertEquals("8", jsonPointer0.getMatchingProperty());
      assertTrue(boolean0);
      assertTrue(jsonPointer0.mayMatchProperty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.matchProperty("Dj");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("\"");
      assertTrue(jsonPointer0.mayMatchProperty());
      
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("");
      assertTrue(jsonPointer1.matches());
      assertEquals("\"", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertNotNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("'?");
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("'?");
      assertEquals("'?", jsonPointer0.toString());
      assertEquals("?", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
      assertNull(jsonPointer1);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.matchElement(2);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("(327");
      JsonPointer jsonPointer1 = jsonPointer0.matchElement(327);
      assertNotNull(jsonPointer1);
      assertEquals("327", jsonPointer0.getMatchingProperty());
      assertTrue(jsonPointer1.matches());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.matchElement((-1));
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("com.faste=xm_.jacks+n.core.JsonPointer");
      boolean boolean0 = jsonPointer0.equals((Object) null);
      assertFalse(boolean0);
      assertEquals("om.faste=xm_.jacks+n.core.JsonPointer", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("com.faste=xm_.jacks+n.core.JsonPointer", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.equals(jsonPointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.equals("m5r z#");
      assertFalse(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      JsonPointer jsonPointer1 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.equals(jsonPointer1);
      assertTrue(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("0914323167");
      JsonPointer jsonPointer1 = new JsonPointer("0914323167", "0914323167", jsonPointer0);
      assertTrue(jsonPointer0.mayMatchElement());
      assertFalse(jsonPointer1.matches());
      assertEquals(914323167, jsonPointer1.getMatchingIndex());
      assertEquals("914323167", jsonPointer0.getMatchingProperty());
      assertEquals("0914323167", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("/bMwi;0#vM8/.V|");
      assertFalse(jsonPointer0.matches());
      assertEquals("bMwi;0#vM8", jsonPointer0.getMatchingProperty());
      assertEquals("/bMwi;0#vM8/.V|", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("rdK;8TPb44b$^J~");
      assertEquals("rdK;8TPb44b$^J~", jsonPointer0.toString());
      assertEquals("dK;8TPb44b$^J~", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("@Kq~", 2);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("@Kq~", jsonPointer0.toString());
      assertEquals("~q~", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail(")\"Vk457'~J~cV3xj/f");
      assertFalse(jsonPointer0.matches());
      assertEquals(")\"Vk457'~J~cV3xj/f", jsonPointer0.toString());
      assertEquals("\"Vk457'~J~cV3xj", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("R;ALR|FM?r(p4'~0");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(jsonPointer0.matches());
      assertEquals("R;ALR|FM?r(p4'~0", jsonPointer0.toString());
      assertEquals("R;ALR|FM?r(p4'~", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("w9{~1^OZ{F\"q=~hL1mk");
      assertEquals("9{/^OZ{F\"q=~hL1mk", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.mayMatchElement());
  }
}