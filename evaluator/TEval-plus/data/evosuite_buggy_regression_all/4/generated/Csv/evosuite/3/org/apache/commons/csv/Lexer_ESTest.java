/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:33:01 GMT 2023
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
      cSVLexer0.readEscape();
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
  public void test01()  throws Throwable  {
      StringReader stringReader0 = new StringReader("j{\"/DCN");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      cSVLexer0.readEscape();
      cSVLexer0.readEscape();
      Token token0 = new Token();
      try { 
        cSVLexer0.nextToken(token0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // (startline 0) EOF reached before encapsulated token finished
         //
         verifyException("org.apache.commons.csv.CSVLexer", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("bN18bwL<:r");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(8, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("fK");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(12, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      StringReader stringReader0 = new StringReader("nnTWLXBc&");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(10, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      StringReader stringReader0 = new StringReader("rk<ij;wt$zHmq");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(13, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StringReader stringReader0 = new StringReader("t=i{RT?D-`/EZs|Aj");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      int int0 = cSVLexer0.readEscape();
      assertEquals(9, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      StringReader stringReader0 = new StringReader("\r\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "\r\n");
      cSVLexer0.trimTrailingSpaces(stringBuilder0);
      assertEquals("", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("dCYIi,A(T3");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      Token token0 = new Token();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token0, token1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("\r\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      Token token0 = new Token();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token1, token0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      StringReader stringReader0 = new StringReader("Q\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      boolean boolean0 = cSVLexer0.readEndOfLine(13);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("j Mu");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      boolean boolean0 = cSVLexer0.isWhitespace(9);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      StringReader stringReader0 = new StringReader("\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVFormat cSVFormat0 = CSVFormat.DEFAULT;
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      cSVLexer0.readEscape();
      Token token0 = new Token();
      stringReader0.reset();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token1, token0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("\r\n");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      cSVLexer0.readEscape();
      Token token0 = new Token();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token0, token1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.RFC4180;
      StringReader stringReader0 = new StringReader("9Gk5,Yxam");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      Token token0 = new Token();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token0, token1);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.MYSQL;
      StringReader stringReader0 = new StringReader("w'v;FwAMY");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      Token token0 = new Token();
      Token token1 = cSVLexer0.nextToken(token0);
      assertSame(token1, token0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CSVFormat cSVFormat0 = CSVFormat.TDF;
      StringReader stringReader0 = new StringReader("c2!i");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      CSVLexer cSVLexer0 = new CSVLexer(cSVFormat0, extendedBufferedReader0);
      boolean boolean0 = cSVLexer0.isCommentStart(65534);
      assertTrue(boolean0);
  }
}