/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:37:19 GMT 2023
 */

package org.apache.commons.math.fraction;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.text.ChoiceFormat;
import java.text.DecimalFormat;
import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParseException;
import java.text.ParsePosition;
import java.util.Locale;
import org.apache.commons.math.fraction.Fraction;
import org.apache.commons.math.fraction.ProperFractionFormat;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ProperFractionFormat_ESTest extends ProperFractionFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat();
      Fraction fraction0 = Fraction.ZERO;
      FieldPosition fieldPosition0 = new FieldPosition(1184);
      // Undeclared exception!
      try { 
        properFractionFormat0.format(fraction0, (StringBuffer) null, fieldPosition0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ChoiceFormat choiceFormat0 = new ChoiceFormat("Ay3pc0L%umd}x-");
      DecimalFormat decimalFormat0 = new DecimalFormat("Ay3pc0L%umd}x-");
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat(decimalFormat0, choiceFormat0, decimalFormat0);
      Fraction fraction0 = properFractionFormat0.parse("Ay3pc0L%umd}x-");
      assertEquals(0, fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat();
      Fraction fraction0 = Fraction.ONE;
      FieldPosition fieldPosition0 = new FieldPosition(1184);
      // Undeclared exception!
      try { 
        properFractionFormat0.format(fraction0, (StringBuffer) null, fieldPosition0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ChoiceFormat choiceFormat0 = new ChoiceFormat("o");
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat(choiceFormat0, choiceFormat0, choiceFormat0);
      Fraction fraction0 = properFractionFormat0.parse("o");
      assertEquals(0.0F, fraction0.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat();
      try { 
        properFractionFormat0.parse("");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Unparseable fraction number: \"\"
         //
         verifyException("org.apache.commons.math.fraction.FractionFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat();
      ParsePosition parsePosition0 = new ParsePosition(1);
      properFractionFormat0.parse("Y4c1u/A{", parsePosition0);
      assertEquals("java.text.ParsePosition[index=1,errorIndex=2]", parsePosition0.toString());
      assertEquals(1, parsePosition0.getIndex());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ChoiceFormat choiceFormat0 = new ChoiceFormat("_OUVM1l)r");
      NumberFormat numberFormat0 = NumberFormat.getInstance();
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat(choiceFormat0, choiceFormat0, numberFormat0);
      try { 
        properFractionFormat0.parse(" / ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Unparseable fraction number: \" / \"
         //
         verifyException("org.apache.commons.math.fraction.FractionFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      ChoiceFormat choiceFormat0 = new ChoiceFormat("ZGF]t)OFA0TM_5PPS7");
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat(choiceFormat0, choiceFormat0, choiceFormat0);
      try { 
        properFractionFormat0.parse("ZGF]t)OFA0TM_5PPS7");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Unparseable fraction number: \"ZGF]t)OFA0TM_5PPS7\"
         //
         verifyException("org.apache.commons.math.fraction.FractionFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(1);
      ChoiceFormat choiceFormat0 = new ChoiceFormat("$Bo~^Q]hDM91',:");
      Locale locale0 = Locale.forLanguageTag("$Bo~^Q]hDM91',:");
      NumberFormat numberFormat0 = NumberFormat.getIntegerInstance(locale0);
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat(numberFormat0, choiceFormat0, choiceFormat0);
      // Undeclared exception!
      try { 
        properFractionFormat0.parse("S1/CbDJx@I~W=", parsePosition0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // The denominator must not be zero
         //
         verifyException("org.apache.commons.math.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = null;
      try {
        properFractionFormat0 = new ProperFractionFormat((NumberFormat) null, (NumberFormat) null, (NumberFormat) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // whole format can not be null.
         //
         verifyException("org.apache.commons.math.fraction.ProperFractionFormat", e);
      }
  }
}
