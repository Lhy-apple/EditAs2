/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:11:53 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Parser_ESTest extends Parser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = Parser.parse("<!P-", "<!P-");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = Parser.parse("<![CDATA[", "<![CDATA[");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = Parser.parse("<html>\n<head>\n</head>\n<body>\n }i`<bgtg>\n  M)j{sKmGw\n </bgtg>\n</body>\n</html><<!-->\n</<!-->", "@w");
      assertEquals("@w", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = Parser.parse("<?", "<?");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("<!--", "<!--");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Parser.parse(";Y?z@c[</", ";Y?z@c[</");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = Parser.parse("<base />\n<html>\n<head>\n</head>\n<body>\n base\n</body>\n</html><base />", "<base />\n<html>\n<head>\n</head>\n<body>\n base\n</body>\n</html><base />");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = Parser.parse("*h-\"B8 %d<hr", "base");
      assertEquals("base", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("U<SA=>P</I", "U<SA=>P</I");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = Parser.parse("<html>\n<head>\n</head>\n<body>\n bys/ \n</body>\n</html><bys/>\n</bys/>", "<html>\n<head>\n</head>\n<body>\n bys/ \n</body>\n</html><bys/>\n</bys/>");
      assertEquals("<html>\n<head>\n</head>\n<body>\n bys/ \n</body>\n</html><bys/>\n</bys/>", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("<html>\n<head>\n <title>}iL$&lt;bGTG,&gt;)jsKqG</title>\n</head>\n<body>\n</body>\n</html>", "<html>\n<head>\n <title>}iL$&lt;bGTG,&gt;)jsKqG</title>\n</head>\n<body>\n</body>\n</html>");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = Parser.parse("<html>\n<head>\n</head>\n<body>\n U<sa p=\"\">\n </sa>\n</body>\n</html>", "<html>\n<head>\n</head>\n<body>\n U<sa p=\"\">\n </sa>\n</body>\n</html>");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = Parser.parse("<x{*3Q=1Aw?G7", "<x{*3Q=1Aw?G7");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = Parser.parse("<xD{*R Q=b A7?", "<xD{*R Q=b A7?");
      assertEquals("#document", document0.nodeName());
  }
}
