/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:19:59 GMT 2023
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
      JsonPointer jsonPointer0 = JsonPointer._parseTail("_@/eG;l)>dw");
      String string0 = jsonPointer0.getMatchingProperty();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("_@/eG;l)>dw", jsonPointer0.toString());
      assertNotNull(string0);
      assertEquals("@", string0);
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.tail();
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("Invalid input: JSON Pointer expression must start with '/': \"", 0);
      String string0 = jsonPointer0.toString();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertNotNull(string0);
      assertEquals("~Invalid input: JSON Pointer expression must start with '", jsonPointer0.getMatchingProperty());
      assertEquals("Invalid input: JSON Pointer expression must start with '/': \"", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("/#");
      assertFalse(jsonPointer0.matches());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("#", jsonPointer0.getMatchingProperty());
      assertEquals("/#", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.mayMatchProperty();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      int int0 = jsonPointer0.getMatchingIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf((String) null);
      assertEquals("", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("");
      assertTrue(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      // Undeclared exception!
      try { 
        JsonPointer.valueOf(".k");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid input: JSON Pointer expression must start with '/': \".k\"
         //
         verifyException("com.fasterxml.jackson.core.JsonPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("eh/fylz!`tqFs3e~`");
      boolean boolean0 = jsonPointer0.matches();
      assertEquals("h", jsonPointer0.getMatchingProperty());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertFalse(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("*4");
      JsonPointer jsonPointer1 = jsonPointer0.matchElement(4);
      assertNotNull(jsonPointer1);
      
      boolean boolean0 = jsonPointer1.matches();
      assertEquals("4", jsonPointer0.getMatchingProperty());
      assertTrue(boolean0);
      assertEquals("*4", jsonPointer0.toString());
      assertTrue(jsonPointer0.mayMatchProperty());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertFalse(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("y6");
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertTrue(boolean0);
      assertFalse(jsonPointer0.matches());
      assertEquals("6", jsonPointer0.getMatchingProperty());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("y6", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.matchProperty(";?xMY%,bD+/:Hr:3");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("@@/eG;l)>Cw");
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("@@/eG;l)>Cw");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertNull(jsonPointer1);
      assertFalse(jsonPointer0.matches());
      assertEquals("@", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      JsonPointer jsonPointer1 = new JsonPointer("", "", jsonPointer0);
      assertTrue(jsonPointer1.mayMatchProperty());
      
      JsonPointer jsonPointer2 = jsonPointer1.matchProperty("");
      assertSame(jsonPointer2, jsonPointer0);
      assertEquals((-1), jsonPointer1.getMatchingIndex());
      assertEquals((-1), jsonPointer2.getMatchingIndex());
      assertNotNull(jsonPointer2);
      assertTrue(jsonPointer2.equals((Object)jsonPointer1));
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.matchElement(925);
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.matchElement((-1));
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("xh @TYz}J$]4");
      JsonPointer jsonPointer1 = new JsonPointer();
      boolean boolean0 = jsonPointer0.equals(jsonPointer1);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals((-1), jsonPointer1.getMatchingIndex());
      assertFalse(jsonPointer1.equals((Object)jsonPointer0));
      assertFalse(jsonPointer0.matches());
      assertEquals("h @TYz}J$]4", jsonPointer0.getMatchingProperty());
      assertEquals("xh @TYz}J$]4", jsonPointer0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.equals(jsonPointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.equals((Object) null);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      Object object0 = new Object();
      boolean boolean0 = jsonPointer0.equals(object0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("61q-y4~");
      assertEquals("1q-y4~", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
      assertFalse(jsonPointer0.mayMatchElement());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("i~P+Zl=~rK");
      assertEquals("~P+Zl=~rK", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("w6'{mUzq~a&~");
      assertEquals("6'{mUzq~a&~", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("w6'{mUzq~a&~", jsonPointer0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("x`04\"fvJ\u0003deP", 2);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(jsonPointer0.matches());
      assertEquals("~4\"fvJ\u0003deP", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("ag~1=\"X'{<Ua_VC3");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("g/=\"X'{<Ua_VC3", jsonPointer0.getMatchingProperty());
      assertEquals("ag~1=\"X'{<Ua_VC3", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
  }
}