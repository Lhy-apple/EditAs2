/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:15:23 GMT 2023
 */

package org.apache.commons.math.complex;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.text.AttributedCharacterIterator;
import java.text.NumberFormat;
import java.text.ParseException;
import java.text.ParsePosition;
import org.apache.commons.math.complex.Complex;
import org.apache.commons.math.complex.ComplexFormat;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ComplexFormat_ESTest extends ComplexFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ComplexFormat complexFormat0 = new ComplexFormat();
      ParsePosition parsePosition0 = new ParsePosition(18);
      parsePosition0.setIndex(0);
      complexFormat0.parse("0-<5##}AQ2", parsePosition0);
      assertEquals(0, parsePosition0.getIndex());
      assertEquals(2, parsePosition0.getErrorIndex());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      String string0 = ComplexFormat.formatComplex(complex0);
      assertEquals("(NaN) + (NaN)i", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
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
  public void test03()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      try { 
        complexFormat0.parseObject("2_|9rW;M");
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
  public void test05()  throws Throwable  {
      ComplexFormat complexFormat0 = new ComplexFormat("overflow: gcd is 2^31");
      Long long0 = new Long(4503599627370496L);
      AttributedCharacterIterator attributedCharacterIterator0 = complexFormat0.formatToCharacterIterator(long0);
      assertEquals(21, attributedCharacterIterator0.getRunLimit());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Complex complex0 = Complex.INF;
      String string0 = ComplexFormat.formatComplex(complex0);
      assertEquals("(Infinity) + (Infinity)i", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      Complex complex0 = complexFormat0.parse("(NaN) + (NaN)i");
      assertEquals(Double.NaN, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      try { 
        complexFormat0.parse(" + ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Unparseable complex number: \" + \"
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      Complex complex0 = complexFormat0.parse("3");
      assertEquals(0.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ComplexFormat complexFormat0 = new ComplexFormat();
      ParsePosition parsePosition0 = new ParsePosition(2);
      complexFormat0.parse(" - ", parsePosition0);
      assertEquals("java.text.ParsePosition[index=2,errorIndex=2]", parsePosition0.toString());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ComplexFormat complexFormat0 = null;
      try {
        complexFormat0 = new ComplexFormat((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // imaginaryCharacter must be a non-empty string.
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ComplexFormat complexFormat0 = ComplexFormat.getInstance();
      // Undeclared exception!
      try { 
        complexFormat0.setImaginaryCharacter("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // imaginaryCharacter must be a non-empty string.
         //
         verifyException("org.apache.commons.math.complex.ComplexFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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