/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:33:36 GMT 2023
 */

package org.apache.commons.math.complex;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.Locale;
import org.apache.commons.math.complex.Complex;
import org.apache.commons.math.complex.ComplexFormat;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ComplexFormat_ESTest extends ComplexFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ComplexFormat complexFormat0 = null;
      try {
        complexFormat0 = new ComplexFormat("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // imaginaryCharacter must be a non-empty string.
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ComplexFormat complexFormat0 = new ComplexFormat();
      Complex complex0 = complexFormat0.parse("7k");
      assertNotNull(complex0);
      assertEquals(7.0, complex0.abs(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Complex complex0 = new Complex(0.0, 691.973636771271);
      String string0 = ComplexFormat.formatComplex(complex0);
      assertEquals("0 + 691.97i", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      NumberFormat numberFormat0 = NumberFormat.getInstance(locale0);
      ComplexFormat complexFormat0 = new ComplexFormat(numberFormat0);
      try { 
        complexFormat0.parseObject("0 + 691.97i");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Format.parseObject(String) failed
         //
         verifyException("java.text.Format", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ComplexFormat complexFormat0 = null;
      try {
        complexFormat0 = new ComplexFormat((NumberFormat) null, (NumberFormat) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // imaginaryFormat can not be null.
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      String string0 = ComplexFormat.formatComplex(complex0);
      assertEquals("(NaN) + (NaN)i", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      Short short0 = new Short((short)800);
      String string0 = complexFormat0.format((Object) short0);
      assertEquals("800", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        ComplexFormat.formatComplex((Complex) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot format given Object as a Date
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      try { 
        complexFormat0.parse(",S{-*rdh8{D&Lr");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Unparseable complex number: \",S{-*rdh8{D&Lr\"
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      try { 
        complexFormat0.parse("9,+M7>(<");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Unparseable complex number: \"9,+M7>(<\"
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ComplexFormat complexFormat0 = new ComplexFormat();
      NumberFormat numberFormat0 = NumberFormat.getCurrencyInstance();
      complexFormat0.setRealFormat(numberFormat0);
      try { 
        complexFormat0.parseObject("-\u00A4741.50 + 1i");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Format.parseObject(String) failed
         //
         verifyException("java.text.Format", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ComplexFormat complexFormat0 = new ComplexFormat();
      Complex complex0 = (Complex)complexFormat0.parseObject("0");
      assertEquals(0.0, complex0.abs(), 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ComplexFormat complexFormat0 = new ComplexFormat();
      try { 
        complexFormat0.parseObject(" ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Format.parseObject(String) failed
         //
         verifyException("java.text.Format", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      // Undeclared exception!
      try { 
        complexFormat0.setImaginaryCharacter((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // imaginaryCharacter must be a non-empty string.
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      // Undeclared exception!
      try { 
        complexFormat0.setRealFormat((NumberFormat) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // realFormat can not be null.
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }
}