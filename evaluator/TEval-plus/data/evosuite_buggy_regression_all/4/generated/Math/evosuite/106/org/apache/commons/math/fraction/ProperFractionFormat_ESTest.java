/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:15:51 GMT 2023
 */

package org.apache.commons.math.fraction;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.text.AttributedCharacterIterator;
import java.text.ChoiceFormat;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.text.ParsePosition;
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
      AttributedCharacterIterator attributedCharacterIterator0 = properFractionFormat0.formatToCharacterIterator(fraction0);
      assertEquals(5, attributedCharacterIterator0.getEndIndex());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ChoiceFormat choiceFormat0 = new ChoiceFormat("Kh");
      DecimalFormat decimalFormat0 = new DecimalFormat();
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat(choiceFormat0, choiceFormat0, decimalFormat0);
      ParsePosition parsePosition0 = new ParsePosition(0);
      properFractionFormat0.parse(" /n", parsePosition0);
      assertEquals("java.text.ParsePosition[index=0,errorIndex=2]", parsePosition0.toString());
      assertEquals(2, parsePosition0.getErrorIndex());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat();
      Fraction fraction0 = Fraction.ONE;
      AttributedCharacterIterator attributedCharacterIterator0 = properFractionFormat0.formatToCharacterIterator(fraction0);
      assertEquals(7, attributedCharacterIterator0.getRunLimit());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat();
      ParsePosition parsePosition0 = new ParsePosition(0);
      properFractionFormat0.parse("0d", parsePosition0);
      assertEquals("java.text.ParsePosition[index=2,errorIndex=-1]", parsePosition0.toString());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat();
      try { 
        properFractionFormat0.parseObject("f-L^tfD}%N");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Format.parseObject(String) failed
         //
         verifyException("java.text.Format", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat();
      ParsePosition parsePosition0 = new ParsePosition(1);
      properFractionFormat0.parse("h8alGK", parsePosition0);
      assertEquals("java.text.ParsePosition[index=1,errorIndex=2]", parsePosition0.toString());
      assertEquals(1, parsePosition0.getIndex());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(1);
      ChoiceFormat choiceFormat0 = new ChoiceFormat("l.G");
      ProperFractionFormat properFractionFormat0 = new ProperFractionFormat(choiceFormat0, choiceFormat0, choiceFormat0);
      properFractionFormat0.parse("l.G", parsePosition0);
      assertEquals("java.text.ParsePosition[index=1,errorIndex=1]", parsePosition0.toString());
      assertEquals(1, parsePosition0.getErrorIndex());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
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