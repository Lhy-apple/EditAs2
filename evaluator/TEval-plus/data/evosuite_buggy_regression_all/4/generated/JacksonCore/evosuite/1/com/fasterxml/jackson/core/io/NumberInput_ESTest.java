/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:34:45 GMT 2023
 */

package com.fasterxml.jackson.core.io;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.io.NumberInput;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberInput_ESTest extends NumberInput_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      char[] charArray0 = new char[1];
      try { 
        NumberInput.parseBigDecimal(charArray0);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.math.BigDecimal", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseLong((char[]) null, (-1), (-1));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      try { 
        NumberInput.parseBigDecimal("*H1MS,D{Q]5rS.");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.math.BigDecimal", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NumberInput numberInput0 = new NumberInput();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      char[] charArray0 = new char[31];
      int int0 = NumberInput.parseInt(charArray0, 0, 0);
      assertEquals((-48), int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      char[] charArray0 = new char[6];
      int int0 = NumberInput.parseInt(charArray0, 3, 2);
      assertEquals((-528), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      char[] charArray0 = new char[8];
      int int0 = NumberInput.parseInt(charArray0, 4, 4);
      assertEquals((-53328), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      char[] charArray0 = new char[17];
      int int0 = NumberInput.parseInt(charArray0, 3, 3);
      assertEquals((-5328), int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      char[] charArray0 = new char[10];
      int int0 = NumberInput.parseInt(charArray0, 5, 5);
      assertEquals((-533328), int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      char[] charArray0 = new char[9];
      int int0 = NumberInput.parseInt(charArray0, 1, 8);
      assertEquals((-533333328), int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      char[] charArray0 = new char[69];
      int int0 = NumberInput.parseInt(charArray0, 6, 6);
      assertEquals((-5333328), int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      char[] charArray0 = new char[34];
      int int0 = NumberInput.parseInt(charArray0, 7, 7);
      assertEquals((-53333328), int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      char[] charArray0 = new char[25];
      int int0 = NumberInput.parseInt(charArray0, 9, 9);
      assertEquals((-1038366032), int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      int int0 = NumberInput.parseInt("5120");
      assertEquals(5120, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      int int0 = NumberInput.parseInt("-3");
      assertEquals((-3), int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("-");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("-'PL|6L@V:%'.`");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-'PL|6L@V:%'.`\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("KNQD+s2Tb=,`");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"KNQD+s2Tb=,`\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseLong(";G");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \";G\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt(")keFD(J8h");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \")keFD(J8h\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("0>?+H_L");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"0>?+H_L\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("7%{%BSg+");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"7%{%BSg+\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      int int0 = NumberInput.parseInt("12");
      assertEquals(12, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("20Ao");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"20Ao\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("20\"MAo");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"20\"MAo\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      int int0 = NumberInput.parseInt("120");
      assertEquals(120, int0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("5120A");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"5120A\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("520+");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"520+\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseLong("2.2250738585072012e-308");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"2.2250738585072012e-308\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      char[] charArray0 = new char[30];
      boolean boolean0 = NumberInput.inLongRange(charArray0, 3, 3, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      char[] charArray0 = new char[25];
      boolean boolean0 = NumberInput.inLongRange(charArray0, 1831, 1831, false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      char[] charArray0 = new char[5];
      charArray0[0] = '9';
      boolean boolean0 = NumberInput.inLongRange(charArray0, (int) '\u0000', 19, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      char[] charArray0 = new char[3];
      charArray0[0] = 'E';
      boolean boolean0 = NumberInput.inLongRange(charArray0, (int) '\u0000', 19, true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      boolean boolean0 = NumberInput.inLongRange("y//MILQ`?mmB>Y=;W", false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      boolean boolean0 = NumberInput.inLongRange("8[.^O(:k6#%bYu7N,sx7", true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      boolean boolean0 = NumberInput.inLongRange("9OS*+sN!6ekh7(%Y'I", true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      boolean boolean0 = NumberInput.inLongRange("8[.^O(:k6#%bYuN,sx7", true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      int int0 = NumberInput.parseAsInt((String) null, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("", 2042);
      assertEquals(2042, int0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("+W+", 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("-I2<(7C", 4);
      assertEquals(4, int0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("8", 50);
      assertEquals(8, int0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      long long0 = NumberInput.parseAsLong((String) null, 234L);
      assertEquals(234L, long0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      long long0 = NumberInput.parseAsLong("", 0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      long long0 = NumberInput.parseAsLong("+K0xt!`", (-1L));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      long long0 = NumberInput.parseAsLong("-'PL|AL@V%'.`", (-1L));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      long long0 = NumberInput.parseAsLong("3", 1535L);
      assertEquals(3L, long0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      double double0 = NumberInput.parseAsDouble("", 0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double double0 = NumberInput.parseAsDouble((String) null, (-1403));
      assertEquals((-1403.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      double double0 = NumberInput.parseAsDouble("MwJx4<:(ZjC7Aa4f}h#", 0L);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("2.2250738585072012e-308", 2029);
      assertEquals(0, int0);
  }
}