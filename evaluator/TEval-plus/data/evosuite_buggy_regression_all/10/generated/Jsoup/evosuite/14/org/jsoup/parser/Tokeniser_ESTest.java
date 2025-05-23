/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:47:50 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.parser.CharacterReader;
import org.jsoup.parser.Token;
import org.jsoup.parser.Tokeniser;
import org.jsoup.parser.TokeniserState;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Tokeniser_ESTest extends Tokeniser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.createTempBuffer();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      // Undeclared exception!
      try { 
        tokeniser0.emitDoctypePending();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.Tokeniser", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.acknowledgeSelfClosingFlag();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.createCommentPending();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("GEwvz,IxSR[l>");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      // Undeclared exception!
      try { 
        tokeniser0.isAppropriateEndTagToken();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.Tokeniser", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("p[.KaSfN,LtD2&-{3Ew");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals("", characterReader0.toString());
      assertEquals("p[.KaSfN,LtD2&-{3Ew", token0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      // Undeclared exception!
      try { 
        tokeniser0.emitTagPending();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.Tokeniser", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("GEwvz,IxSR[l>");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      boolean boolean0 = tokeniser0.isTrackErrors();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      // Undeclared exception!
      try { 
        tokeniser0.emitCommentPending();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.Tokeniser", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Conint");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.setTrackErrors(false);
      tokeniser0.consumeCharacterReference((Character) null, true);
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      boolean boolean0 = tokeniser0.currentNodeInHtmlNS();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.createDoctypePending();
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Ld+%Go^4K");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.getState();
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Fouriertrf");
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag0 = new Token.StartTag("", attributes0);
      token_StartTag0.selfClosing = false;
      token_StartTag0.selfClosing = true;
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.emit(token_StartTag0);
      Token token0 = tokeniser0.read();
      assertSame(token0, token_StartTag0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.emit(token_StartTag0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.emit(token_EndTag0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      Attributes attributes0 = token_EndTag0.attributes;
      attributes0.put("9[-|Uy9*bbq&'^7~", "9[-|Uy9*bbq&'^7~");
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      // Undeclared exception!
      try { 
        tokeniser0.emit(token_EndTag0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.Tokeniser", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("M$|6)`B5Y7]CBqtl&");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals("M$|6)`B5Y7]CBqtl&", token0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Fouriertrf");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = Character.valueOf('C');
      tokeniser0.consumeCharacterReference(character0, true);
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = new Character('#');
      Character character1 = tokeniser0.consumeCharacterReference(character0, true);
      assertNull(character1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&m?pET6Mwr a3Hz3P");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = tokeniser0.consumeCharacterReference((Character) null, true);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#2PNsxa]");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.consumeCharacterReference((Character) null, false);
      assertEquals("PNsxa]", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#XN4`]");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = tokeniser0.consumeCharacterReference((Character) null, false);
      assertNull(character0);
      assertEquals("#XN4`]", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("blockquote");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = tokeniser0.consumeCharacterReference((Character) null, true);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(";KMXP");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = tokeniser0.consumeCharacterReference((Character) null, true);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Fouriertrf");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.consumeCharacterReference((Character) null, false);
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("cir8E");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = new Character('o');
      Character character1 = tokeniser0.consumeCharacterReference(character0, true);
      assertNull(character1);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ges=es");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = tokeniser0.consumeCharacterReference((Character) null, true);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      Token.Tag token_Tag0 = tokeniser0.createTagPending(false);
      assertNotNull(token_Tag0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(")X<i%&#n{");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals(")X", token0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.setTrackErrors(false);
      TokeniserState tokeniserState0 = TokeniserState.AfterDoctypeSystemIdentifier;
      tokeniser0.error(tokeniserState0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(":*%T1J'S");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      TokeniserState tokeniserState0 = TokeniserState.RCDATAEndTagName;
      tokeniser0.error(tokeniserState0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.setTrackErrors(false);
      TokeniserState tokeniserState0 = TokeniserState.ScriptDataEscapedEndTagName;
      tokeniser0.eofError(tokeniserState0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      Attributes attributes0 = token_EndTag0.attributes;
      attributes0.put("9[-|Uy9*bbq&'^7~", "9[-|Uy9*bbq&'^7~");
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.setTrackErrors(false);
      tokeniser0.emit(token_EndTag0);
  }
}
