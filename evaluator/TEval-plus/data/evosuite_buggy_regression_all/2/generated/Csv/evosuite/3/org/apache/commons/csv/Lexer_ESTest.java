/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:49:13 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.IOException;
import java.io.StringReader;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVLexer;
import org.apache.commons.csv.ExtendedBufferedReader;
import org.apache.commons.csv.Quote;
import org.apache.commons.csv.Token;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Lexer_ESTest extends Lexer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      StringReader stringReader0 = new StringReader("\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(10, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      StringReader stringReader0 = new StringReader("");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      long long0 = cSVLexer0.getLineNumber();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      StringReader stringReader0 = new StringReader("");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      try { 
        cSVLexer0.readEscape();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // EOF whilst processing escape sequence
         //
         verifyException("org.apache.commons.csv.Lexer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      StringReader stringReader0 = new StringReader("bM\"9$XF2Do(G");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(8, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StringReader stringReader0 = new StringReader("f}&qn\"M;qXKyHD(\"7");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(12, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      StringReader stringReader0 = new StringReader("npQP?)%aWi+70m&)K6");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(10, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StringReader stringReader0 = new StringReader("r");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(13, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("tR/8E&M#");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(9, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      StringReader stringReader0 = new StringReader("The comment start character and the quoteChar cannot be the same ('");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(84, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      StringReader stringReader0 = new StringReader("\r\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "\r\n");
      cSVLexer0.trimTrailingSpaces(stringBuilder0);
      assertEquals("", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      StringReader stringReader0 = new StringReader("\r\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      boolean boolean0 = cSVLexer0.readEndOfLine(13);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      StringReader stringReader0 = new StringReader("i}r:t17ZFl'8u;Ap>2");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      boolean boolean0 = cSVLexer0.isWhitespace(44);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader(" SurroundingSpaces:ignored");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      Token token0 = new Token();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token0, token1);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      StringReader stringReader0 = new StringReader("\r\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      cSVLexer0.readEscape();
      stringReader0.reset();
      // Undeclared exception!
      try { 
        cSVLexer0.nextToken((Token) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.csv.CSVLexer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      boolean boolean0 = cSVLexer0.isStartOfLine(98);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.EXCEL;
      StringReader stringReader0 = new StringReader("WcAq4M,'4");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      Token token0 = new Token();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token0, token1);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      StringReader stringReader0 = new StringReader("$%&!=UTF.l-aI");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      Token token0 = new Token();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token0, token1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      StringReader stringReader0 = new StringReader("");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      boolean boolean0 = cSVLexer0.isQuoteChar(34);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      StringReader stringReader0 = new StringReader("The comment start character and the quoteChar cannot be the same ('");
      Quote quote0 = Quote.MINIMAL;
      Character character0 = Character.valueOf('T');
      String[] stringArray0 = new String[0];
      CSVFormat cSVFormat0 = new CSVFormat('T', (Character) null, quote0, character0, (Character) null, false, true, "", "The comment start character and the quoteChar cannot be the same ('", stringArray0);
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      boolean boolean0 = cSVLexer0.isCommentStart(84);
      assertTrue(boolean0);
  }
}