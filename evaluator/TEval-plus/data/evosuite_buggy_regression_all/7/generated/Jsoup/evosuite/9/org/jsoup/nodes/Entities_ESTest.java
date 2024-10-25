/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:17:04 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Entities;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Entities_ESTest extends Entities_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Document document0 = new Document("org.jsoup&no_es.Evaluator$ContansOwsText");
      Document.OutputSettings document_OutputSettings0 = document0.outputSettings();
      String string0 = Entities.escape("org.jsoup&no_es.Evaluator$ContansOwsText", document_OutputSettings0);
      assertEquals("org.jsoup&amp;no_es.Evaluator$ContansOwsText", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Entities entities0 = new Entities();
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      String string0 = Entities.unescape(">w/-c&el#24j&a~R");
      assertEquals(">w/-c\u2A99#24j&a~R", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      String string0 = Entities.unescape("w%`#zfQ DE;,#E,@KP");
      assertEquals("w%`#zfQ DE;,#E,@KP", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = Entities.unescape("Aw/-c&#24j&a4~Ru");
      assertEquals("Aw/-c\u0018j&a4~Ru", string0);
  }
}
