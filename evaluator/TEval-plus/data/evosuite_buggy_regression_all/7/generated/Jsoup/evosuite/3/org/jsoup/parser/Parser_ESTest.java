/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:16:00 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Parser_ESTest extends Parser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = Parser.parse("<![CDAcA[", "<![CDAcA[");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = Parser.parse("<![CDATA[", "<![CDATA[");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = Parser.parse("p<FwYj,j<OM+z:", "p<FwYj,j<OM+z:");
      assertEquals("p<FwYj,j<OM+z:", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = Parser.parse("<!--", "<!--");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Parser.parse("<?", "<?");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Parser.parse("</Gy%:lRW4H%U=>'Iq", "</Gy%:lRW4H%U=>'Iq");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = Parser.parse("<!--%s-->", "<!--%s-->");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("</", "</");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("<:4-", "<:4-");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = Parser.parse("<oy%:lRW4H%U=>'Iq", "<oy%:lRW4H%U=>'Iq");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        Parser.parse("qkW<U]=1y.%,B{}2c", "qkW<U]=1y.%,B{}2c");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("drkw<Br$P-#", "drkw<Br$P-#");
      assertEquals("#root", document0.tagName());
  }
}
