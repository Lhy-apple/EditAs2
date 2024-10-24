/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:16:00 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.parser.Token;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Token_ESTest extends Token_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Token.Tag token_Tag0 = token_StartTag0.reset();
      assertSame(token_Tag0, token_StartTag0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      boolean boolean0 = token_EndTag0.isCData();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.tokenType();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      // Undeclared exception!
      try { 
        token_StartTag0.asEndTag();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.jsoup.parser.Token$StartTag cannot be cast to org.jsoup.parser.Token$EndTag
         //
         verifyException("org.jsoup.parser.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Token.CData token_CData0 = new Token.CData("{Z {WL BmoR?");
      Token.Character token_Character0 = token_CData0.asCharacter();
      assertSame(token_Character0, token_CData0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Token.StartTag token_StartTag1 = token_StartTag0.asStartTag();
      assertSame(token_StartTag0, token_StartTag1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      // Undeclared exception!
      try { 
        token_EndTag0.asDoctype();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.jsoup.parser.Token$EndTag cannot be cast to org.jsoup.parser.Token$Doctype
         //
         verifyException("org.jsoup.parser.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Token.CData token_CData0 = new Token.CData("FzU");
      // Undeclared exception!
      try { 
        token_CData0.asComment();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.jsoup.parser.Token$CData cannot be cast to org.jsoup.parser.Token$Comment
         //
         verifyException("org.jsoup.parser.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      Attributes attributes0 = token_EndTag0.getAttributes();
      assertNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      String string0 = token_StartTag0.normalName();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      char[] charArray0 = new char[8];
      token_EndTag0.appendAttributeValue(charArray0);
      token_EndTag0.appendAttributeName('I');
      token_EndTag0.finaliseTag();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.name("SOI3G`");
      String string0 = token_StartTag0.toString();
      assertEquals("<SOI3G`>", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendTagName('7');
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      boolean boolean0 = token_EndTag0.isSelfClosing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      // Undeclared exception!
      try { 
        token_EndTag0.toString();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be false
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Token.EOF token_EOF0 = new Token.EOF();
      Token token0 = token_EOF0.reset();
      assertSame(token0, token_EOF0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      String string0 = token_Comment0.toString();
      assertEquals("<!---->", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      Token token0 = token_Comment0.reset();
      assertSame(token_Comment0, token0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Token.CData token_CData0 = new Token.CData((String) null);
      String string0 = token_CData0.toString();
      assertEquals("<![CDATA[null]]>", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Token.CData token_CData0 = new Token.CData("{Z {WL BmoR?");
      Token token0 = token_CData0.reset();
      Token.TokenType token_TokenType0 = Token.TokenType.EOF;
      token0.type = token_TokenType0;
      boolean boolean0 = token_CData0.isEOF();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Token.Character token_Character0 = new Token.Character();
      String string0 = token_Character0.toString();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.getSystemIdentifier();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.getName();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      Token.Doctype token_Doctype1 = (Token.Doctype)token_Doctype0.reset();
      Token.TokenType token_TokenType0 = Token.TokenType.EndTag;
      token_Doctype1.type = token_TokenType0;
      boolean boolean0 = token_Doctype1.isDoctype();
      assertFalse(token_Doctype0.isForceQuirks());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.getPubSysKey();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.getPublicIdentifier();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      boolean boolean0 = token_Doctype0.isForceQuirks();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Token.reset((StringBuilder) null);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.newAttribute();
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.appendAttributeName("           ");
      token_StartTag0.newAttribute();
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      char[] charArray0 = new char[0];
      token_EndTag0.appendAttributeValue(charArray0);
      token_EndTag0.appendAttributeName('7');
      token_EndTag0.finaliseTag();
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeName('H');
      token_EndTag0.setEmptyAttributeValue();
      token_EndTag0.newAttribute();
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.finaliseTag();
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.nameAttr("", (Attributes) null);
      // Undeclared exception!
      try { 
        token_StartTag0.toString();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be false
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Token.Tag token_Tag0 = token_StartTag0.name("7mNWI)hma<O");
      token_StartTag0.appendTagName("]");
      assertSame(token_StartTag0, token_Tag0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeName('H');
      token_EndTag0.appendAttributeName('>');
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeValue('\\');
      token_EndTag0.appendAttributeValue("");
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      int[] intArray0 = new int[2];
      token_EndTag0.appendAttributeValue(intArray0);
      assertEquals(2, intArray0.length);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.appendAttributeValue("-!(~.0!Xq%h;P@");
      char[] charArray0 = new char[2];
      token_StartTag0.appendAttributeValue(charArray0);
      assertArrayEquals(new char[] {'\u0000', '\u0000'}, charArray0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.appendAttributeName("AfterAttributeValue_quoted");
      token_StartTag0.newAttribute();
      // Undeclared exception!
      try { 
        token_StartTag0.toString();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be false
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      boolean boolean0 = token_Doctype0.isDoctype();
      assertFalse(token_Doctype0.isForceQuirks());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      boolean boolean0 = token_EndTag0.isStartTag();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      boolean boolean0 = token_StartTag0.isStartTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      boolean boolean0 = token_StartTag0.isEndTag();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      boolean boolean0 = token_EndTag0.isEndTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Token.CData token_CData0 = new Token.CData((String) null);
      boolean boolean0 = token_CData0.isComment();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      Token.TokenType token_TokenType0 = Token.TokenType.Comment;
      token_EndTag0.type = token_TokenType0;
      boolean boolean0 = token_EndTag0.isComment();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Token.EOF token_EOF0 = new Token.EOF();
      boolean boolean0 = token_EOF0.isCharacter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Token.CData token_CData0 = new Token.CData((String) null);
      boolean boolean0 = token_CData0.isCharacter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      boolean boolean0 = token_StartTag0.isEOF();
      assertFalse(boolean0);
  }
}
