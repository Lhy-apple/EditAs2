/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:03:27 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
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
      CharacterReader characterReader0 = new CharacterReader("");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
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
      CharacterReader characterReader0 = new CharacterReader("");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.acknowledgeSelfClosingFlag();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.createCommentPending();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("nexist");
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
      CharacterReader characterReader0 = new CharacterReader("&#S3wNg+}(i");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals("", characterReader0.toString());
      assertEquals("&#S3wNg+}(i", token0.toString());
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
      CharacterReader characterReader0 = new CharacterReader("uzFRP9[C M}u");
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
      CharacterReader characterReader0 = new CharacterReader("nexist");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.setTrackErrors(false);
      tokeniser0.consumeCharacterReference((Character) null, true);
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("LpjZ<jH[M5 p{^X%foz");
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
      CharacterReader characterReader0 = new CharacterReader("vi6ScF?jVe(sw\"4#=");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.getState();
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.selfClosing = true;
      tokeniser0.emit(token_StartTag0);
      // Undeclared exception!
      try { 
        tokeniser0.read();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.Tokeniser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.emit(token_EndTag0);
      Token token0 = tokeniser0.read();
      assertSame(token0, token_EndTag0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      Token.StartTag token_StartTag0 = new Token.StartTag();
      tokeniser0.emit(token_StartTag0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeName('S');
      token_EndTag0.newAttribute();
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
      CharacterReader characterReader0 = new CharacterReader("vs|vUobVo)&");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals("vs|vUobVo)&", token0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("ic2H%%)6u6)Vz8");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = new Character('{');
      Character character1 = tokeniser0.consumeCharacterReference(character0, true);
      assertNull(character1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("kfr");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = new Character('k');
      Character character1 = tokeniser0.consumeCharacterReference(character0, true);
      assertNull(character1);
      assertEquals("kfr", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("<Dot");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = tokeniser0.consumeCharacterReference((Character) null, true);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#x4#cDx!'lg<v");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.consumeCharacterReference((Character) null, false);
      assertEquals("#cDx!'lg<v", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("#4#cDx!'lg<v");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.consumeCharacterReference((Character) null, false);
      assertEquals("#cDx!'lg<v", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("bG&Mg,H[]9F},9`Q");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals("bG&Mg,H[]9F},9`Q", token0.toString());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("HQ;Fbb>(fgB");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = tokeniser0.consumeCharacterReference((Character) null, true);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("nrtri;");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = new Character('E');
      tokeniser0.consumeCharacterReference(character0, false);
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("nexs");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Character character0 = tokeniser0.consumeCharacterReference((Character) null, true);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("nexist");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.consumeCharacterReference((Character) null, true);
      assertEquals("", characterReader0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("^L</T2pHcf24cQ");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals("^L", token0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("sXC*JmLE7<y~b");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals("sXC*JmLE7", token0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("uzFRP9[C M}u");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      tokeniser0.setTrackErrors(false);
      TokeniserState tokeniserState0 = TokeniserState.DoctypeSystemIdentifier_doubleQuoted;
      tokeniser0.error(tokeniserState0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      CharacterReader characterReader0 = new CharacterReader("p+%N.B#B<`y8G");
      Tokeniser tokeniser0 = new Tokeniser(characterReader0);
      Token token0 = tokeniser0.read();
      assertEquals("p+%N.B#B<`y8G", token0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.setTrackErrors(false);
      TokeniserState tokeniserState0 = TokeniserState.ScriptDataEscaped;
      tokeniser0.eofError(tokeniserState0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Token.EndTag token_EndTag0 = new Token.EndTag();
      token_EndTag0.appendAttributeName('S');
      token_EndTag0.newAttribute();
      Tokeniser tokeniser0 = new Tokeniser((CharacterReader) null);
      tokeniser0.setTrackErrors(false);
      tokeniser0.emit(token_EndTag0);
  }
}
