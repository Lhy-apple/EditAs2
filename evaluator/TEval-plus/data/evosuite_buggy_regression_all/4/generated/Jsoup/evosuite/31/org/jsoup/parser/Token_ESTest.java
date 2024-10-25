/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:51:20 GMT 2023
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
      Token.StartTag token_StartTag0 = new Token.StartTag((String) null);
      token_StartTag0.appendTagName('>');
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.tokenType();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("</");
      Token.EndTag token_EndTag1 = token_EndTag0.asEndTag();
      Token.TokenType token_TokenType0 = Token.TokenType.Comment;
      token_EndTag1.type = token_TokenType0;
      boolean boolean0 = token_EndTag1.isComment();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      // Undeclared exception!
      try { 
        token_Comment0.asCharacter();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.jsoup.parser.Token$Comment cannot be cast to org.jsoup.parser.Token$Character
         //
         verifyException("org.jsoup.parser.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag("org.jsoup.parser.Token$Doctype", (Attributes) null);
      Token.StartTag token_StartTag1 = token_StartTag0.asStartTag();
      assertSame(token_StartTag0, token_StartTag1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("9#gQr");
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
  public void test06()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag("org.jsoup.parser.Token$Doctype", (Attributes) null);
      // Undeclared exception!
      try { 
        token_StartTag0.asComment();
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.jsoup.parser.Token$StartTag cannot be cast to org.jsoup.parser.Token$Comment
         //
         verifyException("org.jsoup.parser.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("Pattern syntax error: ");
      Attributes attributes0 = token_EndTag0.getAttributes();
      assertNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("org.jsoup.nodes.Attribute");
      Token.Tag token_Tag0 = token_EndTag0.name(":<kR");
      assertSame(token_EndTag0, token_Tag0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("Pattern syntax error: ");
      boolean boolean0 = token_EndTag0.isSelfClosing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Token.EOF token_EOF0 = new Token.EOF();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      String string0 = token_Comment0.toString();
      assertEquals("<!---->", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.getSystemIdentifier();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.getName();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      token_Doctype0.getPublicIdentifier();
      assertFalse(token_Doctype0.isForceQuirks());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      boolean boolean0 = token_Doctype0.isForceQuirks();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Token.Character token_Character0 = new Token.Character("");
      String string0 = token_Character0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.newAttribute();
      token_EndTag0.newAttribute();
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeValue("Comment");
      token_EndTag0.appendAttributeName('z');
      token_EndTag0.newAttribute();
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag((String) null);
      token_StartTag0.finaliseTag();
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("org.jsoup.nodes.Attribute");
      token_EndTag0.appendAttributeName("org.jsoup.nodes.Attribute");
      token_EndTag0.finaliseTag();
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("");
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
  public void test22()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("</");
      token_EndTag0.appendTagName("");
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeName('z');
      token_EndTag0.appendAttributeName("EndTag");
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeValue("Comment");
      token_EndTag0.appendAttributeValue('.');
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag("5zJN8P");
      token_StartTag0.attributes = null;
      String string0 = token_StartTag0.toString();
      assertEquals("<5zJN8P>", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.newAttribute();
      Attributes attributes0 = token_EndTag0.attributes;
      Token.StartTag token_StartTag0 = new Token.StartTag("=Q'!o6Tr'%4]vz", attributes0);
      String string0 = token_StartTag0.toString();
      assertEquals("<=Q'!o6Tr'%4]vz>", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeName('8');
      token_EndTag0.newAttribute();
      Attributes attributes0 = token_EndTag0.attributes;
      Token.StartTag token_StartTag0 = new Token.StartTag("=Q'!o6Tr'%4]vz", attributes0);
      String string0 = token_StartTag0.toString();
      assertEquals("<=Q'!o6Tr'%4]vz  8=\"\">", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      boolean boolean0 = token_Comment0.isDoctype();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      Token.TokenType token_TokenType0 = Token.TokenType.Doctype;
      token_Comment0.type = token_TokenType0;
      boolean boolean0 = token_Comment0.isDoctype();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      boolean boolean0 = token_Comment0.isStartTag();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      Token.TokenType token_TokenType0 = Token.TokenType.Comment;
      token_Comment0.type = token_TokenType0;
      Token.TokenType token_TokenType1 = Token.TokenType.StartTag;
      token_Comment0.type = token_TokenType1;
      boolean boolean0 = token_Comment0.isStartTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag((String) null);
      boolean boolean0 = token_StartTag0.isEndTag();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag("");
      boolean boolean0 = token_EndTag0.isEndTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag((String) null);
      boolean boolean0 = token_StartTag0.isComment();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag((String) null);
      boolean boolean0 = token_StartTag0.isCharacter();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Token.Character token_Character0 = new Token.Character("");
      boolean boolean0 = token_Character0.isCharacter();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      boolean boolean0 = token_Comment0.isEOF();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      Token.TokenType token_TokenType0 = Token.TokenType.EOF;
      token_Comment0.type = token_TokenType0;
      boolean boolean0 = token_Comment0.isEOF();
      assertTrue(boolean0);
  }
}
