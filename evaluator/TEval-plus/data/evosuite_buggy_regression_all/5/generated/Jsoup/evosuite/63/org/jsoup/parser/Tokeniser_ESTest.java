/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:16:45 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.parser.CharacterReader;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.Token;
import org.jsoup.parser.Tokeniser;
import org.jsoup.parser.TokeniserState;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Tokeniser_ESTest extends Tokeniser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("$I^ukwQ\"F.fS]\"k:)4");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, (ParseErrorList) null);
      tokeniser0.createTempBuffer();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("");
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(1099);
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      tokeniser0.emitDoctypePending();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&#xa04aF;");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      tokeniser0.acknowledgeSelfClosingFlag();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null, parseErrorList0);
      tokeniser0.createCommentPending();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      CharacterReader characterReader0 = new CharacterReader("&gt;");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token token0 = tokeniser0.read();
      assertEquals(4, characterReader0.pos());
      assertEquals(">", token0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null, parseErrorList0);
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
  public void test06()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IkBm4zv&sYc$6#'H");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token token0 = tokeniser0.read();
      assertTrue(characterReader0.isEmpty());
      assertEquals("IkBm4zv&sYc$6#'H", token0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("LkB4hv&sc$6#'H");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      tokeniser0.emitCommentPending();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&#x304aF;");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      boolean boolean0 = tokeniser0.currentNodeInHtmlNS();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ZnmEL");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      tokeniser0.createDoctypePending();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ParseErrorList parseErrorList0 = new ParseErrorList(0, 0);
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null, parseErrorList0);
      tokeniser0.getState();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IkBm4zv&sYc&6#'H");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      // Undeclared exception!
      try { 
        tokeniser0.emit((char[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("IkBm4zv&sYc$6#'H");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      tokeniser0.emit(token_StartTag0);
      Token.StartTag token_StartTag1 = (Token.StartTag)tokeniser0.read();
      token_StartTag1.selfClosing = true;
      tokeniser0.emit(token_StartTag1);
      tokeniser0.read();
      assertTrue(parseErrorList0.isEmpty());
      assertEquals(0, parseErrorList0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Unexpected character '%s' in input state [%s]");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token.EndTag token_EndTag0 = new Token.EndTag();
      tokeniser0.emit(token_EndTag0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("Unexpected character '%s' in input state [%s]");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.newAttribute();
      tokeniser0.emit(token_EndTag0);
      assertTrue(parseErrorList0.isEmpty());
      assertEquals(0, parseErrorList0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("[oL,+(G5ku[m:j`&");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(true);
      assertTrue(characterReader0.isEmpty());
      assertEquals("[oL,+(G5ku[m:j`&", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("-xNM@8mljj]/v|");
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(1021);
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Character character0 = new Character('o');
      tokeniser0.consumeCharacterReference(character0, false);
      assertTrue(parseErrorList0.isEmpty());
      assertEquals(0, parseErrorList0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("],D9x -$$XAD&#6*}");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Character character0 = new Character(']');
      int[] intArray0 = tokeniser0.consumeCharacterReference(character0, false);
      assertNull(intArray0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("& L#x3K4aF;");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(false);
      assertTrue(characterReader0.isEmpty());
      assertEquals("& L#x3K4aF;", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&#xa8EaFa");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(false);
      assertEquals(9, characterReader0.pos());
      assertEquals("\uFFFD", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("qCxL -P$yGXD&#6*d}");
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(83);
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(true);
      assertTrue(characterReader0.isEmpty());
      assertEquals("qCxL -P$yGXD\u0006*d}", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      CharacterReader characterReader0 = new CharacterReader("q9xL r-P$yGXAf&#}");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(true);
      assertEquals(17, characterReader0.pos());
      assertEquals("q9xL r-P$yGXAf&#}", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&#xa04aF;");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, (ParseErrorList) null);
      String string0 = tokeniser0.unescapeEntities(false);
      assertTrue(characterReader0.isEmpty());
      assertEquals("\uDA41\uDCAF", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("LkB4hv&sc$6#'H");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(false);
      assertTrue(characterReader0.isEmpty());
      assertEquals("LkB4hv&sc$6#'H", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&t;");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(true);
      assertTrue(characterReader0.isEmpty());
      assertEquals("&t;", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("4&gt-~z6~EB");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(true);
      assertEquals(11, characterReader0.pos());
      assertEquals("4&gt-~z6~EB", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("&gt;");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(true);
      assertTrue(characterReader0.isEmpty());
      assertEquals(">", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      CharacterReader characterReader0 = new CharacterReader("&gt~3;");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      String string0 = tokeniser0.unescapeEntities(false);
      assertEquals(6, characterReader0.pos());
      assertEquals(">~3;", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("o0+.u7?>Yt6dq%|");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token.Tag token_Tag0 = tokeniser0.createTagPending(false);
      assertNotNull(token_Tag0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("numeric reference with no numerals");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token.StartTag token_StartTag0 = (Token.StartTag)tokeniser0.createTagPending(true);
      Attributes attributes0 = new Attributes();
      token_StartTag0.nameAttr("numeric reference with no numerals", attributes0);
      tokeniser0.emit(token_StartTag0);
      String string0 = tokeniser0.appropriateEndTagName();
      assertEquals("numeric reference with no numerals", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("tPli{");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      boolean boolean0 = tokeniser0.isAppropriateEndTagToken();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("L0Bhv&sc$6#'a");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token.StartTag token_StartTag0 = (Token.StartTag)tokeniser0.createTagPending(true);
      Attributes attributes0 = new Attributes();
      token_StartTag0.nameAttr("L0Bhv&sc$6#'a", attributes0);
      tokeniser0.emit(token_StartTag0);
      boolean boolean0 = tokeniser0.isAppropriateEndTagToken();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("T");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, (ParseErrorList) null);
      String string0 = tokeniser0.appropriateEndTagName();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      CharacterReader characterReader0 = new CharacterReader("<+0oC");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token token0 = tokeniser0.read();
      assertTrue(parseErrorList0.isEmpty());
      assertEquals("<+0oC", token0.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("-xNM@8mljj]/v|");
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(1021);
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      TokeniserState tokeniserState0 = TokeniserState.ScriptDataDoubleEscapeEnd;
      tokeniser0.error(tokeniserState0);
      assertFalse(parseErrorList0.isEmpty());
      assertEquals(1, parseErrorList0.size());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader(">,@fk5g;w1fz]^?<h");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token token0 = tokeniser0.read();
      assertEquals(">,@fk5g;w1fz]^?", token0.toString());
      assertTrue(parseErrorList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("tPli{");
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(55);
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      TokeniserState tokeniserState0 = TokeniserState.CharacterReferenceInData;
      tokeniser0.eofError(tokeniserState0);
      assertEquals(1, parseErrorList0.size());
      assertFalse(parseErrorList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(1099);
      CharacterReader characterReader0 = new CharacterReader("numeric reference with no numerals");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      tokeniser0.error("");
      assertFalse(parseErrorList0.isEmpty());
      assertEquals(1, parseErrorList0.size());
  }
}