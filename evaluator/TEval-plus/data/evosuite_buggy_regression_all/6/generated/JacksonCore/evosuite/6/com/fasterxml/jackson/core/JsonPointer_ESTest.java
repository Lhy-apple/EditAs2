/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:18:01 GMT 2023
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
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("z^#9q#sw", 0);
      String string0 = jsonPointer0.getMatchingProperty();
      assertEquals("~z^#9q#sw", string0);
      assertEquals("z^#9q#sw", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile((String) null);
      JsonPointer jsonPointer1 = jsonPointer0.tail();
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.toString();
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.valueOf("");
      boolean boolean0 = jsonPointer0.equals(jsonPointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      int int0 = jsonPointer0.getMatchingIndex();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      try { 
        JsonPointer.compile("1621547712");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid input: JSON Pointer expression must start with '/': \"1621547712\"
         //
         verifyException("com.fasterxml.jackson.core.JsonPointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile("/J?+WK");
      assertEquals("/J?+WK", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
      assertEquals("J?+WK", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("=WQi?");
      boolean boolean0 = jsonPointer0.matches();
      assertEquals("=WQi?", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("WQi?", jsonPointer0.getMatchingProperty());
      assertFalse(boolean0);
      assertTrue(jsonPointer0.mayMatchProperty());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      boolean boolean0 = jsonPointer0.matches();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("1621547712");
      boolean boolean0 = jsonPointer0.mayMatchProperty();
      assertEquals("1621547712", jsonPointer0.toString());
      assertTrue(boolean0);
      assertEquals(1621547712, jsonPointer0.getMatchingIndex());
      assertEquals("1621547712", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertFalse(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("1621547712");
      boolean boolean0 = jsonPointer0.mayMatchElement();
      assertTrue(boolean0);
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("1621547712", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.compile("");
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("");
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("%g>'D,/9", 1);
      jsonPointer0.matchProperty("%g>'D,/9");
      assertEquals("%g>'D,/9", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("~g>'D,", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("Value \"", 1);
      JsonPointer jsonPointer1 = jsonPointer0.matchProperty("~alue \"");
      assertNotNull(jsonPointer1);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("", jsonPointer1.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer.EMPTY;
      JsonPointer jsonPointer1 = jsonPointer0.matchElement(70);
      assertNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("58");
      JsonPointer jsonPointer1 = jsonPointer0.matchElement(58);
      assertEquals("58", jsonPointer0.toString());
      assertEquals("58", jsonPointer0.getMatchingProperty());
      assertNotNull(jsonPointer1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      jsonPointer0.matchElement((-1));
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("^82`M");
      boolean boolean0 = jsonPointer0.equals("^82`M");
      assertEquals("^82`M", jsonPointer0.toString());
      assertEquals("82`M", jsonPointer0.getMatchingProperty());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertFalse(jsonPointer0.matches());
      assertFalse(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      boolean boolean0 = jsonPointer0.equals((Object) null);
      assertFalse(boolean0);
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JsonPointer jsonPointer0 = new JsonPointer();
      JsonPointer jsonPointer1 = new JsonPointer("2.2250738585072012e-308", "2.2250738585072012e-308", jsonPointer0);
      boolean boolean0 = jsonPointer0.equals(jsonPointer1);
      assertFalse(jsonPointer1.matches());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(boolean0);
      assertTrue(jsonPointer1.mayMatchProperty());
      assertEquals((-1), jsonPointer1.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("1");
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("1", jsonPointer0.toString());
      assertFalse(jsonPointer0.matches());
      assertEquals("", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("(+~0");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("+~", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("7621547718");
      assertEquals("7621547718", jsonPointer0.toString());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("7621547718", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("%g>'D,/9");
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(jsonPointer0.matches());
      assertEquals("g>'D,", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("k6v00,M=Wrer_O~");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("k6v00,M=Wrer_O~", jsonPointer0.toString());
      assertTrue(jsonPointer0.mayMatchProperty());
      assertEquals("6v00,M=Wrer_O~", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("=r<aR0:[pPK :{~:Y~");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertFalse(jsonPointer0.matches());
      assertEquals("r<aR0:[pPK :{~:Y~", jsonPointer0.getMatchingProperty());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseQuotedTail("~F~A=", 1);
      assertEquals("~F~A=", jsonPointer0.getMatchingProperty());
      assertEquals((-1), jsonPointer0.getMatchingIndex());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JsonPointer jsonPointer0 = JsonPointer._parseTail("76~154718");
      assertEquals((-1), jsonPointer0.getMatchingIndex());
      assertEquals("76~154718", jsonPointer0.toString());
      assertEquals("76/54718", jsonPointer0.getMatchingProperty());
      assertFalse(jsonPointer0.matches());
  }
}
