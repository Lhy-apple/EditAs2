/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:45:10 GMT 2023
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
      char[] charArray0 = new char[4];
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
      char[] charArray0 = new char[0];
      // Undeclared exception!
      try { 
        NumberInput.parseLong(charArray0, 2260, 2260);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 2260
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      try { 
        NumberInput.parseBigDecimal("");
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
      char[] charArray0 = new char[17];
      int int0 = NumberInput.parseInt(charArray0, 0, 0);
      assertEquals((-48), int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      char[] charArray0 = new char[8];
      int int0 = NumberInput.parseInt(charArray0, 0, 7);
      assertEquals((-53333328), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      char[] charArray0 = new char[30];
      int int0 = NumberInput.parseInt(charArray0, 0, 2);
      assertEquals((-528), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      char[] charArray0 = new char[8];
      int int0 = NumberInput.parseInt(charArray0, 0, 3);
      assertEquals((-5328), int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      char[] charArray0 = new char[17];
      int int0 = NumberInput.parseInt(charArray0, 4, 4);
      assertEquals((-53328), int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      char[] charArray0 = new char[17];
      int int0 = NumberInput.parseInt(charArray0, 5, 5);
      assertEquals((-533328), int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      char[] charArray0 = new char[8];
      int int0 = NumberInput.parseInt(charArray0, 1, 6);
      assertEquals((-5333328), int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      char[] charArray0 = new char[8];
      int int0 = NumberInput.parseInt(charArray0, 0, 8);
      assertEquals((-533333328), int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      char[] charArray0 = new char[17];
      int int0 = NumberInput.parseInt(charArray0, 0, 12);
      assertEquals((-1038366032), int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      int int0 = NumberInput.parseInt("1762");
      assertEquals(1762, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("-byRZJ-e9K_^,'clvf");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-byRZJ-e9K_^,'clvf\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
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
        NumberInput.parseInt("-45@");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-45@\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("WL#Yf:|htWLVj");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"WL#Yf:|htWLVj\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("-KS9d");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-KS9d\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt(" ");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \" \"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      int int0 = NumberInput.parseInt("2");
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("7= $z~");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"7= $z~\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("9)GX)r");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"9)GX)r\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      int int0 = NumberInput.parseInt("-45");
      assertEquals((-45), int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("17(2'xJ");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"17(2'xJ\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      int int0 = NumberInput.parseInt("172");
      assertEquals(172, int0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("172v'");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"172v'\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseInt("1762'");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"1762'\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseLong("}?X.4Klp%9h");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"}?X.4Klp%9h\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberInput.parseLong("");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      char[] charArray0 = new char[9];
      boolean boolean0 = NumberInput.inLongRange(charArray0, (int) '\u0000', (int) '\u0000', false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      char[] charArray0 = new char[17];
      boolean boolean0 = NumberInput.inLongRange(charArray0, 1783, 1783, true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      char[] charArray0 = new char[1];
      boolean boolean0 = NumberInput.inLongRange(charArray0, 0, 19, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      char[] charArray0 = new char[1];
      charArray0[0] = '9';
      // Undeclared exception!
      try { 
        NumberInput.inLongRange(charArray0, 0, 19, true);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      char[] charArray0 = new char[3];
      charArray0[0] = 'L';
      boolean boolean0 = NumberInput.inLongRange(charArray0, 0, 19, false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      boolean boolean0 = NumberInput.inLongRange("Qb/$^Kb3MmL6#ePg>t", true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      boolean boolean0 = NumberInput.inLongRange("com.fasterxml.jackson.core.io.NumberInput", false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      boolean boolean0 = NumberInput.inLongRange(",]BDN?!DkP^?*b}M|", false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("", (-533328));
      assertEquals((-533328), int0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      int int0 = NumberInput.parseAsInt((String) null, 2886);
      assertEquals(2886, int0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("+'W~T", 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("2", 72);
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      int int0 = NumberInput.parseAsInt("-KS9d", 5013);
      assertEquals(5013, int0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      long long0 = NumberInput.parseAsLong((String) null, 373L);
      assertEquals(373L, long0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      long long0 = NumberInput.parseAsLong("", 374L);
      assertEquals(374L, long0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      long long0 = NumberInput.parseAsLong("+2", 1);
      assertEquals(2L, long0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      long long0 = NumberInput.parseAsLong("-KS9d", (-3624L));
      assertEquals((-3624L), long0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      long long0 = NumberInput.parseAsLong("2.2250738585072012e-308", 1L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double double0 = NumberInput.parseAsDouble("x x/z>L", 0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      double double0 = NumberInput.parseAsDouble((String) null, 1.0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      double double0 = NumberInput.parseAsDouble("", 0.0);
      assertEquals(0.0, double0, 0.01);
  }
}